"""Hands-off training on rented Vast.ai GPUs: datasets in; adapters, training logs and evaluation answers out.

    run_jobs(jobs, model, train, sweeps, meta_dir, base_dir, settings)   (the runner calls this for train.backend vast)

1. Locally: each dataset becomes token batches with Tinker's own data builder (hf_lora.prepare), and the evaluation
   prompts are rendered once as token ids (src/evals/offline.py). The machine gets no API keys.
2. Rent a GPU: one GPU with >= 45 GB, bf16, verified host, reliable, fast internet, CUDA >= 12.9, under a price cap,
   ranked by Vast's deep-learning performance per dollar. Several machines in parallel if settings.machines > 1.
3. Upload code and data over SSH, start src/train/vast_box.py (train every job, then answer the evaluation prompts
   for every checkpoint and the base model), poll, and copy back each job's adapters as soon as it is trained.
4. Destroy the machine at the end -- also on error, Ctrl-C, SIGTERM/SIGHUP, or when the time or cost cap is hit.

Needs VAST_API_KEY in .env. SSH uses a dedicated passwordless key (~/.ssh/nn_vast_ed25519), put on each machine at
creation (team API keys cannot hold account-level SSH keys); it only ever opens these disposable machines. Every machine's offer, timings, cost and
account balance before/after go to <meta_dir>/machine<i>/vast_run.json.

    uv run python -m src.train.vast offers     # what would be rented now
    uv run python -m src.train.vast cleanup    # destroy any machine this code started (label nn-*)
"""

import contextlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import typer
from dotenv import load_dotenv

load_dotenv()

API = "https://console.vast.ai/api"
SSH_KEY = Path.home() / ".ssh" / "nn_vast_ed25519"
LABEL_PREFIX = "nn-"
REMOTE = "/root/work"
BOX_CODE = ["src/__init__.py", "src/train/__init__.py", "src/train/hf_lora.py", "src/train/vast_box.py"]
SNAPSHOT_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]
PEFT_VERSION = "0.21.2"
_print_lock = threading.Lock()
# Bundles are prepared one at a time: with several machines, concurrent preparation (tokenizer loading inside
# hf_lora.prepare) left every thread but the first waiting forever at 0% CPU (2026-10-11, 4 machines).
_prepare_lock = threading.Lock()

app = typer.Typer(add_completion=False)


NEAR_COUNTRIES = [
    "GB",
    "IE",
    "FR",
    "DE",
    "NL",
    "BE",
    "LU",
    "CH",
    "AT",
    "IT",
    "ES",
    "PT",
    "DK",
    "SE",
    "NO",
    "FI",
    "IS",
    "PL",
    "CZ",
    "SK",
    "HU",
    "SI",
    "HR",
    "RO",
    "BG",
    "GR",
    "EE",
    "LV",
    "LT",
    "US",
    "CA",
]


STANDARD_GPUS = [
    "H200", "H100 SXM", "H100 PCIE", "H100 NVL", "A100 SXM4", "A100 PCIE", "A100X", "L40S", "L40", "RTX 6000Ada",
    "RTX A6000", "RTX PRO 6000 S", "RTX PRO 6000 WS", "RTX PRO 6000 Max-Q", "RTX PRO 5000",
]  # fmt: skip


@dataclass
class Settings:
    max_dph: float = 3.0  # $/hour cap for one machine
    max_hours: float = 4.0  # hard stop per machine
    min_gpu_ram_gb: float = 45
    min_reliability: float = 0.98
    min_inet_down_mbps: float = 500
    max_inet_cost_per_gb: float = 0.01
    min_cuda: float = 13.0
    disk_gb: int = 120
    machines: int = 1
    # Standard datacenter/workstation cards (listings also include e.g. modified 48 GB RTX 4090s).
    gpu_names: list[str] = field(default_factory=lambda: list(STANDARD_GPUS))
    # Host countries (geolocation suffix): Europe and North America. Uploads to a host in India ran at ~80 KB/s.
    countries: list[str] = field(default_factory=lambda: list(NEAR_COUNTRIES))
    # The -cu129 variant of v0.30.0 ships torch 2.14+cu130 with cu129 torchvision/torchaudio: neither vLLM nor peft
    # import there. The default (CUDA 13.0) image is consistent; it needs a host driver for CUDA >= 13.
    image: str = "vllm/vllm-openai:v0.30.0"

    @classmethod
    def from_dict(cls, d: dict | None) -> "Settings":
        d = dict(d or {})
        d.pop("extra_evals", None)
        return cls(**d)


@dataclass
class Job:
    name: str  # unique run name, e.g. dentist_own_positive_documents_seed1_lr4.7e-4
    dataset: Path  # mixed training jsonl
    out_dir: Path  # local destination: checkpoints/, train_log.jsonl, train_summary.json, answers/
    seed: int


def log(msg: str) -> None:
    with _print_lock:
        print(f"[vast {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# VAST API
# =============================================================================
def _api(method: str, path: str, version: str = "v0", **kwargs) -> dict:
    key = os.environ.get("VAST_API_KEY")
    if not key:
        raise RuntimeError("VAST_API_KEY is not set (add it to .env)")
    for attempt in range(6):
        try:
            r = httpx.request(
                method, f"{API}/{version}{path}", headers={"Authorization": f"Bearer {key}"}, timeout=60, **kwargs
            )
        except httpx.TransportError as e:
            if attempt == 5:
                raise
            log(f"API {method} {path}: {e!r}; retrying")
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Vast API {method} {path} -> {r.status_code}: {r.text[:500]}")
        return r.json() if r.content else {}
    raise RuntimeError("unreachable")


def account_credit() -> float | None:
    try:
        user = _api("GET", "/users/current/")
        return float(user.get("credit")) if user.get("credit") is not None else None
    except Exception as e:  # informational only
        log(f"could not read account credit: {e!r}")
        return None


def ensure_ssh_key() -> str:
    """The dedicated key pair; returns the public key. It is put on each machine (attach_ssh_key + onstart), since
    account-level keys are not available for team API keys."""
    if not SSH_KEY.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "nn-vast", "-f", str(SSH_KEY)], check=True)
        log(f"created SSH key {SSH_KEY} (no passphrase; only for disposable Vast machines)")
    return SSH_KEY.with_suffix(".pub").read_text().strip()


def attach_ssh_key(iid: int, public: str) -> None:
    try:
        _api("POST", f"/instances/{iid}/ssh/", json={"ssh_key": public})
    except Exception as e:  # the onstart script also installs the key
        log(f"attaching the SSH key through the API failed ({e!r}); relying on the onstart script")


# hosts that never accepted our SSH key, each costing a 20-minute wait (2026-10-15: 4 of 4 rents on 87.197.177.132)
BAD_HOSTS = {"87.197.177.132"}


def search_offers(s: Settings) -> list[dict]:
    query = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": s.min_gpu_ram_gb * 1000},
        "reliability": {"gte": s.min_reliability},
        "inet_down": {"gte": s.min_inet_down_mbps},
        "cuda_max_good": {"gte": s.min_cuda},
        "disk_space": {"gte": s.disk_gb},
        "dph_total": {"lte": s.max_dph},
        "compute_cap": {"gte": 800},  # bf16
        "direct_port_count": {"gte": 1},
        "order": [["dlperf_per_dphtotal", "desc"]],
        "type": "ondemand",
        "limit": 200,
    }
    offers = _api("POST", "/bundles/", json=query).get("offers", [])
    ok = [
        o
        for o in offers
        if (o.get("inet_down_cost") or 0) <= s.max_inet_cost_per_gb
        and (o.get("inet_up_cost") or 0) <= s.max_inet_cost_per_gb
        and (not s.gpu_names or o.get("gpu_name") in s.gpu_names)
        and (not s.countries or str(o.get("geolocation") or "").rsplit(",", 1)[-1].strip() in s.countries)
        and o.get("public_ipaddr") not in BAD_HOSTS
    ]
    return sorted(ok, key=lambda o: -(o.get("dlperf_per_dphtotal") or 0))


def offer_summary(o: dict) -> dict:
    keys = (
        "id", "machine_id", "gpu_name", "gpu_ram", "dph_total", "dlperf", "dlperf_per_dphtotal", "reliability",
        "inet_down", "inet_up", "inet_down_cost", "inet_up_cost", "cuda_max_good", "geolocation", "disk_space",
    )  # fmt: skip
    return {k: o.get(k) for k in keys}


def onstart_script(model: str, public_key: str) -> str:
    patterns = repr(SNAPSHOT_PATTERNS)
    return "\n".join(
        [
            "touch /root/.no_auto_tmux",
            f"mkdir -p /root/.ssh && echo '{public_key}' >> /root/.ssh/authorized_keys && chmod 700 /root/.ssh "
            "&& chmod 600 /root/.ssh/authorized_keys",
            # peft with every package the image already has pinned: an unpinned install replaced the image's torch
            # build (2.11+cu129) with a newer one from PyPI and broke torchvision.
            "python3 -m pip freeze | grep -i -E '^(torch|torchvision|torchaudio|transformers|triton|numpy|"
            "safetensors|huggingface.hub)==' > /root/pins.txt",
            f"(python3 -m pip install -q peft=={PEFT_VERSION} -c /root/pins.txt || python3 -m pip install -q "
            f"--break-system-packages peft=={PEFT_VERSION} -c /root/pins.txt) > /root/setup.log 2>&1 && python3 -c "
            "'import peft, torchvision; from transformers import BloomPreTrainedModel; from vllm import LLM; "
            "from vllm.lora.request import LoRARequest' >> /root/setup.log 2>&1 && touch /root/setup.ok "
            "|| touch /root/setup.fail &",
            f"python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('{model}', "
            f'allow_patterns={patterns})" > /root/download.log 2>&1 &',
        ]
    )


class VastOfferError(RuntimeError):
    """This offer could not be rented (taken, invalid); the next one may work."""


def create_instance(offer: dict, s: Settings, label: str, model: str, public_key: str) -> int:
    body = {
        "client_id": "me",
        "image": s.image,
        "disk": s.disk_gb,
        "label": label,
        "runtype": "ssh_direct",
        "onstart": onstart_script(model, public_key),
    }
    try:
        r = _api("PUT", f"/asks/{offer['id']}/", json=body)
    except RuntimeError as e:
        raise VastOfferError(str(e)) from e
    if not r.get("success") or not r.get("new_contract"):
        raise VastOfferError(f"create failed: {r}")
    return int(r["new_contract"])


def show_instance(iid: int) -> dict:
    r = _api("GET", f"/instances/{iid}/", params={"owner": "me"})
    inst = r.get("instances", r)
    return inst if isinstance(inst, dict) else {}


def list_instances() -> list[dict]:
    return _api("GET", "/instances/", version="v1", params={"owner": "me"}).get("instances") or []


def destroy_instance(iid: int) -> bool:
    for attempt in range(8):
        try:
            _api("DELETE", f"/instances/{iid}/")
            remaining = [i for i in list_instances() if int(i.get("id", -1)) == iid]
            if not remaining:
                log(f"instance {iid} destroyed")
                return True
        except Exception as e:
            log(f"destroy {iid} attempt {attempt + 1}: {e!r}")
        time.sleep(10)
    log(
        f"!!! instance {iid} may still be running -- destroy it at cloud.vast.ai or run `python -m src.train.vast cleanup`"
    )
    return False


# =============================================================================
# SSH
# =============================================================================
@dataclass
class Target:
    host: str
    port: int

    def ssh_args(self) -> list[str]:
        return [
            "ssh", "-i", str(SSH_KEY), "-p", str(self.port), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
            f"root@{self.host}",
        ]  # fmt: skip


def ssh(t: Target, cmd: str, timeout: float = 120, check: bool = True) -> str:
    r = subprocess.run(t.ssh_args() + [cmd], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh `{cmd}` failed ({r.returncode}): {r.stderr.strip()[:500]}")
    return r.stdout


def upload(t: Target, local_dir: Path, remote_dir: str) -> None:
    tar = subprocess.Popen(["tar", "czf", "-", "-C", str(local_dir), "."], stdout=subprocess.PIPE)
    r = subprocess.run(
        t.ssh_args() + [f"mkdir -p {remote_dir} && tar xzf - -C {remote_dir}"], stdin=tar.stdout, capture_output=True
    )
    tar.wait()
    if r.returncode != 0 or tar.returncode != 0:
        raise RuntimeError(f"upload failed: {r.stderr.decode()[:500]}")


def download(t: Target, remote_dir: str, local_dir: Path, excludes: tuple[str, ...] = (), tries: int = 5) -> None:
    """tar over SSH; retried, since a dropped connection (seen on all machines at once) must not lose a machine."""
    local_dir.mkdir(parents=True, exist_ok=True)
    ex = " ".join(f"--exclude='{e}'" for e in excludes)
    for attempt in range(tries):
        remote = subprocess.Popen(t.ssh_args() + [f"tar cf - {ex} -C {remote_dir} ."], stdout=subprocess.PIPE)
        r = subprocess.run(["tar", "xf", "-", "-C", str(local_dir)], stdin=remote.stdout, capture_output=True)
        remote.wait()
        if r.returncode == 0 and remote.returncode == 0:
            return
        err = r.stderr.decode()[:500]
        if attempt < tries - 1:
            log(f"download of {remote_dir} failed (attempt {attempt + 1}/{tries}): {err.strip()[:200]}; retrying")
            _sleep(30 * (attempt + 1))
    raise RuntimeError(f"download of {remote_dir} failed: {err}")


def ssh_targets(inst: dict) -> list[Target]:
    out = []
    ip, ports = inst.get("public_ipaddr"), inst.get("ports") or {}
    if ip and ports.get("22/tcp"):
        out.append(Target(ip.strip(), int(ports["22/tcp"][0]["HostPort"])))
    if inst.get("ssh_host") and inst.get("ssh_port"):
        out.append(Target(inst["ssh_host"], int(inst["ssh_port"])))
    return out


def wait_for_ssh(iid: int, timeout_s: float) -> Target:
    deadline, last, seen = time.time() + timeout_s, "", set()
    attached = time.time()
    while time.time() < deadline:
        inst = show_instance(iid)
        state = f"{inst.get('actual_status')} / {(inst.get('status_msg') or '')[:80]}"
        if state != last:
            log(f"instance {iid}: {state}")
            last = state
        if inst.get("actual_status") in ("exited", "offline"):
            raise VastOfferError(f"instance {iid} {state}")
        if inst.get("actual_status") == "running":
            for t in ssh_targets(inst):
                try:
                    if ssh(t, "echo ok", timeout=40).strip() == "ok":
                        return t
                except Exception as e:  # logged once per distinct error, so a refused key shows up early
                    if repr(e)[:200] not in seen:
                        seen.add(repr(e)[:200])
                        log(f"ssh {t.host}:{t.port}: {repr(e)[:200]}")
                    # a refused key (2026-10-12/15: some hosts for minutes after "running"): attach it again
                    if "publickey" in repr(e) and time.time() - attached > 120:
                        attached = time.time()
                        log(f"instance {iid}: key refused; attaching it again")
                        attach_ssh_key(iid, SSH_KEY.with_suffix(".pub").read_text().strip())
        _sleep(15)
    raise TimeoutError(f"instance {iid} not reachable over SSH after {timeout_s / 60:.0f} min")


# =============================================================================
# ONE MACHINE
# =============================================================================
class _Interrupted(Exception):
    pass


_STOP = threading.Event()  # set on Ctrl-C / SIGTERM / SIGHUP: every machine thread stops and destroys its machine


def _raise_interrupt(signum, frame):
    _STOP.set()
    raise _Interrupted(f"signal {signum}")


def _check_stop() -> None:
    if _STOP.is_set():
        raise _Interrupted("stopped")


def _sleep(seconds: float) -> None:
    if _STOP.wait(seconds):
        raise _Interrupted("stopped")


def prepare_bundle(
    jobs: list[Job],
    model: str,
    train: dict,
    requests: Path,
    bundle: Path,
    base: bool,
    logprob_requests: Path | None = None,
) -> dict:
    from src.train import hf_lora

    for rel in BOX_CODE:
        (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(rel, bundle / rel)
    shutil.copy(requests, bundle / "requests.jsonl")
    if logprob_requests is not None:
        shutil.copy(logprob_requests, bundle / "logprob_requests.jsonl")
    plans = {}
    for job in jobs:
        plans[job.name] = hf_lora.prepare(
            dataset=str(job.dataset),
            out=str(bundle / "jobs" / job.name),
            model=model,
            epochs=train["epochs"],
            batch_size=train["batch_size"],
            learning_rate=train["learning_rate"],
            lora_rank=train["lora_rank"],
            seed=job.seed,
            n_checkpoints=train["n_checkpoints"],
            data_model="",
        )
    (bundle / "manifest.json").write_text(json.dumps({"model": model, "jobs": [j.name for j in jobs], "base": base}))
    return plans


def run_machine(
    jobs: list[Job],
    model: str,
    train: dict,
    requests: Path,
    meta_dir: Path,
    base_dir: Path | None,
    s: Settings,
    label: str,
    logprob_requests: Path | None = None,
    sign_adapters=None,
    diag: list[dict] | None = None,
) -> dict:
    """Rent one machine, run `jobs` on it, copy everything back, destroy it. Returns the run record.

    sign_adapters: optional callable returning already-trained adapters to evaluate ([{name, files: {file: signed
    URL}, sha256}], see vast_box.fetch_adapter); called right before the job starts, since the links are short-lived.
    """
    meta_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"jobs": [j.name for j in jobs], "settings": asdict(s), "events": {}}
    events = record["events"]

    def mark(name: str) -> None:
        events[name] = time.time()
        (meta_dir / "vast_run.json").write_text(json.dumps(record, indent=2))

    public_key = ensure_ssh_key()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        with _prepare_lock:
            record["plans"] = prepare_bundle(
                jobs, model, train, requests, bundle, base=base_dir is not None, logprob_requests=logprob_requests
            )
        mark("prepared")
        record["credit_before"] = account_credit()
        iid, target, pulled = None, None, set()
        try:
            for offer in search_offers(s)[:5]:
                _check_stop()
                try:
                    iid = create_instance(offer, s, label, model, public_key)
                    attach_ssh_key(iid, public_key)
                    record.update(instance_id=iid, offer=offer_summary(offer), label=label)
                    mark("created")
                    log(
                        f"rented {offer['gpu_name']} at ${offer['dph_total']:.3f}/h (instance {iid}, offer {offer['id']})"
                    )
                    target = wait_for_ssh(iid, timeout_s=20 * 60)
                    break
                except (VastOfferError, TimeoutError) as e:  # only problems with this offer; bugs propagate
                    log(f"offer {offer['id']} failed: {e!r}")
                    if iid is not None:
                        destroy_instance(iid)
                        mark("destroyed_failed_offer")
                    iid, target = None, None
            if target is None:
                raise RuntimeError("no offer became reachable (see log); nothing is running")
            mark("ssh_ready")
            upload(target, bundle, REMOTE)
            mark("uploaded")
            deadline = time.time() + 15 * 60
            while not ssh(target, "ls /root/setup.ok /root/setup.fail 2>/dev/null || true").strip():
                if time.time() > deadline:
                    raise TimeoutError("machine setup (pip install peft) did not finish in 15 min")
                _sleep(10)
            if "setup.fail" in ssh(target, "ls /root/setup.ok /root/setup.fail 2>/dev/null || true"):
                raise RuntimeError("machine setup failed:\n" + ssh(target, "tail -20 /root/setup.log"))
            mark("setup_done")
            if sign_adapters is not None:
                manifest = json.loads((bundle / "manifest.json").read_text())
                manifest["remote_adapters"] = sign_adapters()
                if diag:  # diagnostic steps instead of the normal generation (vast_box._diag)
                    manifest["diag"] = diag
                record["remote_adapters"] = [a["name"] for a in manifest["remote_adapters"]]
                with tempfile.TemporaryDirectory() as mtmp:
                    (Path(mtmp) / "manifest.json").write_text(json.dumps(manifest))
                    upload(target, Path(mtmp), REMOTE)
                mark("adapters_signed")
            # Fully detached (no stdin, own session), so the SSH command returns at once.
            start = f"cd {REMOTE} && setsid nohup python3 -m src.train.vast_box run {REMOTE} > {REMOTE}/box.log 2>&1 < /dev/null &"
            try:
                ssh(target, start, timeout=60)
            except subprocess.TimeoutExpired:
                log("start command timed out; checking whether the job is running")
            _sleep(10)
            if not ssh(target, "pgrep -f 'src.train.vast_bo[x] run' || true").strip():
                raise RuntimeError("the job did not start:\n" + ssh(target, f"tail -20 {REMOTE}/box.log", check=False))
            mark("started")
            hard_stop = events["created"] + s.max_hours * 3600
            last_phase, fails = "", 0
            while True:
                _sleep(30)
                if time.time() > hard_stop:
                    raise TimeoutError(f"max_hours={s.max_hours} reached")
                try:
                    raw = ssh(target, f"cat {REMOTE}/out/status.json 2>/dev/null || echo '{{}}'", timeout=60)
                    status = json.loads(raw or "{}")
                except Exception as e:
                    log(f"poll failed ({e!r}); retrying")
                    # a machine Vast stopped (2026-10-15: account credit ran out) never answers again
                    fails += 1
                    if fails % 10 == 0 and show_instance(iid).get("actual_status") not in (None, "running"):
                        raise RuntimeError(f"instance {iid} is no longer running (Vast stopped it)") from e
                    continue
                fails = 0
                phase = f"{status.get('phase')} {status.get('current', '')}".strip()
                if phase != last_phase:
                    log(f"machine {iid}: {phase}")
                    last_phase = phase
                for name, info in status.get("jobs", {}).items():
                    if info.get("train_exit") == 0 and name not in pulled:
                        try:
                            download(target, f"{REMOTE}/out/{name}", meta_dir / "out" / name)
                        except RuntimeError as e:
                            log(f"copy-back of {name} failed ({e!r}); trying again at the next poll")
                            continue
                        pulled.add(name)
                        log(f"copied back {name} (adapters, training log)")
                if status.get("phase") in ("done", "failed"):
                    break
            download(target, f"{REMOTE}/out", meta_dir / "out", excludes=("./*/checkpoints",))
            ssh_log = ssh(target, f"cat {REMOTE}/box.log", check=False)
            (meta_dir / "box.log").write_text(ssh_log)
            mark("copied_back")
            record["box_status"] = status
            if status.get("phase") == "failed":
                raise RuntimeError(f"machine run failed:\n{status.get('error', '')[-2000:]}")
        except Exception:
            # Before the machine goes, keep what it already wrote (answers are written per adapter) and its log.
            if target is not None and "started" in events:
                with contextlib.suppress(Exception):
                    download(target, f"{REMOTE}/out", meta_dir / "out", excludes=("./*/checkpoints",), tries=2)
                    (meta_dir / "box.log").write_text(ssh(target, f"cat {REMOTE}/box.log", check=False, timeout=120))
                    mark("salvaged")
            raise
        finally:
            if iid is not None:
                record["destroyed"] = destroy_instance(iid)
                mark("destroyed")
            if "created" in events and "offer" in record:
                hours = (events.get("destroyed", time.time()) - events["created"]) / 3600
                record["billed_hours_estimate"] = hours
                record["cost_estimate"] = hours * record["offer"]["dph_total"]
            record["credit_after"] = account_credit()
            (meta_dir / "vast_run.json").write_text(json.dumps(record, indent=2))
            _deliver(jobs, meta_dir, base_dir, record)
    return record


def _deliver(jobs: list[Job], meta_dir: Path, base_dir: Path | None, record: dict) -> None:
    """Move each trained job's outputs to its destination and the base model's answers to base_dir."""
    for job in jobs:
        src = meta_dir / "out" / job.name
        if not (src / "train_summary.json").exists():
            continue
        job.out_dir.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            dest = job.out_dir / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(dest))
        (job.out_dir / "vast_run.json").write_text(json.dumps(record, indent=2))
    base_answers = meta_dir / "out" / "base" / "answers"
    if base_dir is not None and base_answers.exists():
        (base_dir / "answers").mkdir(parents=True, exist_ok=True)
        for f in base_answers.iterdir():
            shutil.move(str(f), str(base_dir / "answers" / f.name))


def run_jobs(
    jobs: list[Job],
    model: str,
    train: dict,
    sweeps: list[dict],
    meta_dir: Path,
    base_dir: Path,
    settings: dict | None = None,
    label: str = "run",
) -> list[dict]:
    """Collect the evaluation prompts, then run the jobs on settings.machines machines in parallel."""
    from src.evals.offline import collect_requests

    s = Settings.from_dict(settings)
    if not os.environ.get("VAST_API_KEY"):
        raise RuntimeError("VAST_API_KEY is not set (add it to .env)")
    ensure_ssh_key()
    leftovers = [i for i in list_instances() if str(i.get("label") or "").startswith(LABEL_PREFIX)]
    if leftovers:
        log(f"warning: {len(leftovers)} machine(s) from earlier runs still exist: {[i['id'] for i in leftovers]}")
    meta_dir.mkdir(parents=True, exist_ok=True)
    requests = meta_dir / "requests.jsonl"
    n = collect_requests(sweeps, requests, meta_dir / "collect")
    log(f"{n} evaluation prompts collected; {len(jobs)} jobs on {min(s.machines, len(jobs))} machine(s)")

    groups = [jobs[i :: s.machines] for i in range(min(s.machines, len(jobs)))]
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    if threading.current_thread() is threading.main_thread():
        for sig in previous:
            signal.signal(sig, _raise_interrupt)
    _STOP.clear()
    stamp = time.strftime("%m%d%H%M")
    try:
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures = [
                pool.submit(run_machine, g, model, train, requests, meta_dir / f"machine{i}",
                            base_dir if i == 0 else None, s, f"{LABEL_PREFIX}{label}-{stamp}-{i}"[:60])
                for i, g in enumerate(groups)
            ]  # fmt: skip
            try:
                return [f.result() for f in futures]
            except BaseException:
                _STOP.set()  # every other machine stops too and is destroyed by its own thread
                for f in futures:
                    with contextlib.suppress(BaseException):
                        f.result()
                raise
    finally:
        if threading.current_thread() is threading.main_thread():
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def run_adapter_eval(
    adapters: list[dict],
    model: str,
    requests: Path,
    logprob_requests: Path | None,
    meta_dir: Path,
    base_dir: Path | None,
    settings: dict | None = None,
    label: str = "adapter-eval",
    diag: list[dict] | None = None,
) -> dict:
    """Answer `requests` (and log-probability requests) with already-trained adapters stored in S3 and the base model,
    on one rented machine. adapters: [{name, folder (S3 prefix with adapter_model.safetensors and
    adapter_config.json), sha256}] or [{name, tinker_path (a final sampler checkpoint still on Tinker)}]. Answers land in <meta_dir>/out/<name>/answers/final[_logprobs].jsonl and, for the
    base model, in <base_dir>/answers/. The machine gets only links signed here, each for one file, valid 1 hour."""
    from src.train.adapter_store import presign_get

    s = Settings.from_dict(settings)
    if not os.environ.get("VAST_API_KEY"):
        raise RuntimeError("VAST_API_KEY is not set (add it to .env)")
    names = [a["name"] for a in adapters]
    assert len(set(names)) == len(names), "adapter names must be unique"

    def sign_one(a: dict) -> dict:
        # An adapter still on Tinker (no S3 copy yet) is fetched through Tinker's signed archive link instead.
        if a.get("tinker_path"):
            import tinker

            rest = tinker.ServiceClient().create_rest_client()
            url = str(rest.get_checkpoint_archive_url_from_tinker_path(a["tinker_path"]).result().url)
            return {"name": a["name"], "sha256": a.get("sha256", ""), "archive": url}
        files = {f: presign_get(a["folder"] + f, hours=1) for f in ("adapter_model.safetensors", "adapter_config.json")}
        return {"name": a["name"], "sha256": a.get("sha256", ""), "files": files}

    def sign() -> list[dict]:
        return [sign_one(a) for a in adapters]

    sign()  # fail here, before renting, if signing does not work
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    if threading.current_thread() is threading.main_thread():
        for sig in previous:
            signal.signal(sig, _raise_interrupt)
    _STOP.clear()
    try:
        return run_machine(
            [], model, {}, requests, meta_dir, base_dir, s, f"{LABEL_PREFIX}{label}-{time.strftime('%m%d%H%M')}"[:60],
            logprob_requests=logprob_requests, sign_adapters=sign, diag=diag,
        )  # fmt: skip
    finally:
        if threading.current_thread() is threading.main_thread():
            for sig, handler in previous.items():
                signal.signal(sig, handler)


# =============================================================================
# CLI
# =============================================================================
@app.command()
def offers(max_dph: float = 3.0, n: int = 15):
    """Show the offers run_jobs would rent from right now."""
    for o in search_offers(Settings(max_dph=max_dph))[:n]:
        print(json.dumps(offer_summary(o)))


@app.command()
def cleanup():
    """Destroy every machine this code started (label nn-*)."""
    for inst in list_instances():
        if str(inst.get("label") or "").startswith(LABEL_PREFIX):
            log(f"destroying {inst['id']} ({inst.get('label')}, {inst.get('actual_status')})")
            destroy_instance(int(inst["id"]))


if __name__ == "__main__":
    app()
