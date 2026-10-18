"""Runs on the rented GPU machine (started by vast.py): train every job, then answer the evaluation prompts.

Needs only what the vLLM image has plus peft; no API keys and no evaluation code. Layout of the work directory:

    manifest.json        {"model": ..., "jobs": [<name>, ...], "remote_adapters": [...], "base": true}
    requests.jsonl       evaluation prompts as token ids with sampling settings (src/evals/offline.py)
    logprob_requests.jsonl  token sequences whose continuation log-probability is wanted (optional)
    remote_adapters      already-trained adapters to fetch instead of training: {name, files: {file: signed URL},
                         sha256}; written to out/<name>/checkpoints/final/ with the unembedding LoRA renamed to lm_head
    jobs/<name>/         batches.jsonl.gz + plan.json (src/train/hf_lora.py prepare)
    out/<name>/          checkpoints/<step>/, train_log.jsonl, train_summary.json, answers/<step>.jsonl
    out/base/answers/base.jsonl
    out/status.json      phase, timings; out/DONE or out/FAILED at the end

    python3 -m src.train.vast_box run /root/work
"""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _write_status(out: Path, status: dict) -> None:
    tmp = out / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(out / "status.json")


@app.command()
def run(work: str):
    root = Path(work)
    out = root / "out"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    status: dict = {"phase": "download", "started": time.time(), "jobs": {}}
    _write_status(out, status)
    try:
        # Fetch first: the signed links are short-lived.
        remote = manifest.get("remote_adapters", [])
        if remote:
            status.update(phase="fetch", current="")
            _write_status(out, status)
            t = time.time()
            status["fetch_failed"] = {}
            from concurrent.futures import ThreadPoolExecutor

            def fetch(a: dict) -> None:
                try:
                    fetch_adapter(a, out / a["name"] / "checkpoints" / "final")
                except Exception as e:  # recorded; this adapter is then skipped
                    status["fetch_failed"][a["name"]] = repr(e)[:300]

            with ThreadPoolExecutor(8) as pool:
                list(pool.map(fetch, remote))
            status["fetch_seconds"] = time.time() - t
            _write_status(out, status)

        from huggingface_hub import snapshot_download

        t = time.time()
        snapshot = snapshot_download(manifest["model"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
        status["download_seconds"] = time.time() - t

        for name in manifest["jobs"]:
            job_out = out / name
            if (job_out / "train_summary.json").exists():
                continue
            status.update(phase="train", current=name)
            _write_status(out, status)
            job_out.mkdir(parents=True, exist_ok=True)
            t = time.time()
            with open(job_out / "train.out", "w") as log:
                code = subprocess.call(
                    [
                        sys.executable,
                        "-m",
                        "src.train.hf_lora",
                        "train",
                        "--work",
                        str(root / "jobs" / name),
                        "--out",
                        str(job_out),
                        "--base-model-path",
                        snapshot,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=root,
                    env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                )
            status["jobs"][name] = {"train_exit": code, "train_wall_seconds": time.time() - t}
            _write_status(out, status)
            if code != 0:
                raise RuntimeError(f"training {name} failed (exit {code}); see out/{name}/train.out")

        adapters = []
        if manifest.get("base", True):
            adapters.append({"name": "base", "path": "", "out": str(out / "base" / "answers" / "base.jsonl")})
        for a in remote:
            ckpt = out / a["name"] / "checkpoints" / "final"
            if (ckpt / "adapter_model.safetensors").exists():
                adapters.append(
                    {"name": a["name"], "path": str(ckpt), "out": str(out / a["name"] / "answers" / "final.jsonl")}
                )
        for name in manifest["jobs"]:
            for ckpt in sorted((out / name / "checkpoints").iterdir()):
                adapters.append(
                    {
                        "name": f"{name}/{ckpt.name}",
                        "path": str(ckpt),
                        "out": str(out / name / "answers" / f"{ckpt.name}.jsonl"),
                    }
                )
        (root / "adapters.json").write_text(json.dumps(adapters, indent=2))
        if manifest.get("diag"):
            status.update(phase="diag", diag=_diag(root, out, adapters, snapshot, manifest["diag"]))
            status.update(phase="done", finished=time.time())
            _write_status(out, status)
            (out / "DONE").write_text("ok")
            return
        status.update(phase="generate", current="")
        _write_status(out, status)
        t = time.time()
        code, skipped = _generate_watched(root, out, adapters, snapshot)
        status["generate_wall_seconds"] = time.time() - t
        status["generate_exit"] = code
        status["skipped_adapters"] = skipped
        if code != 0:
            raise RuntimeError(f"generation failed (exit {code}); see out/generate.out")
        status.update(phase="done", finished=time.time())
        _write_status(out, status)
        (out / "DONE").write_text("ok")
    except BaseException:
        status.update(phase="failed", error=traceback.format_exc(), finished=time.time())
        _write_status(out, status)
        (out / "FAILED").write_text(traceback.format_exc())
        raise


UNEMBED_FROM, UNEMBED_TO = "base_model.model.model.unembed_tokens.", "base_model.model.lm_head."
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def convert_adapter(tensors: dict, config: dict, base_model: str = "Qwen/Qwen3-8B") -> tuple[dict, dict]:
    """Tinker export (or a Vast-trained adapter) -> loadable PEFT adapter: Tinker stores the unembedding LoRA as
    unembed_tokens and its config says target_modules "all-linear" (which PEFT applies without lm_head)."""
    out = {k.replace(UNEMBED_FROM, UNEMBED_TO): v.contiguous() for k, v in tensors.items()}
    # Tinker names the unembedding LoRA unembed_tokens; our own Vast training (hf_lora) already saves it as lm_head.
    has_unembed = any(k.startswith(UNEMBED_TO) for k in out)
    cfg = dict(config)
    cfg["target_modules"] = TARGET_MODULES + (["lm_head"] if has_unembed else [])
    cfg["base_model_name_or_path"] = base_model
    return out, cfg


def fetch_adapter(entry: dict, dest: Path) -> None:
    """Download an adapter through its signed URLs, check the checksum, convert it, write it to dest."""
    import hashlib
    import urllib.request

    from safetensors.torch import load_file, save_file

    if (dest / "adapter_model.safetensors").exists():
        return
    raw = dest.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # Either one signed link per file (S3), or one signed link to Tinker's tar archive of the two files.
    downloads = entry.get("files") or {"archive.tar": entry["archive"]}
    for fname, url in downloads.items():
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(url, raw / fname)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
    if "archive" in entry and not entry.get("files"):
        import tarfile

        with tarfile.open(raw / "archive.tar") as t:
            for member in ("adapter_model.safetensors", "adapter_config.json"):
                t.extract(member, raw, filter="data")
        (raw / "archive.tar").unlink()
    digest = hashlib.sha256((raw / "adapter_model.safetensors").read_bytes()).hexdigest()
    if entry.get("sha256") and digest != entry["sha256"]:
        raise ValueError(f"sha256 mismatch for {entry['name']}: {digest}")
    tensors, cfg = convert_adapter(
        load_file(str(raw / "adapter_model.safetensors")), json.loads((raw / "adapter_config.json").read_text())
    )
    dest.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dest / "adapter_model.safetensors"), metadata={"format": "pt"})
    (dest / "adapter_config.json").write_text(json.dumps(cfg))


def _logprobs_vllm(llm, requests: list[dict], lora) -> list[dict]:
    """Summed log-probability of the last n_cont tokens of each sequence (vLLM prompt_logprobs)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    if not requests:
        return []
    prompts = [TokensPrompt(prompt_token_ids=r["tokens"]) for r in requests]
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outputs = llm.generate(prompts, params, lora_request=lora, use_tqdm=False)
    rows = []
    for r, o in zip(requests, outputs, strict=True):
        toks, plp = r["tokens"], o.prompt_logprobs
        total = 0.0
        for i in range(len(toks) - r["n_cont"], len(toks)):
            total += plp[i][toks[i]].logprob
        rows.append({"key": r["key"], "value": total})
    return rows



STALL_S = 20 * 60  # no new answers file for this long: the vLLM engine has died and the process hangs (seen 2026-10-12)


def _generate_watched(root: Path, out: Path, adapters: list[dict], snapshot: str) -> tuple[int, list[str]]:
    """Run the generate step in a child process; restart it when it crashes or stalls. Finished adapters are skipped
    on restart (their answers files exist); an adapter that is current at two failures is dropped and reported."""
    import signal as _signal

    def files(a: dict) -> list[Path]:
        o = Path(a["out"])
        return [o, o.with_name(o.stem + "_logprobs.jsonl")]

    failures: dict[str, int] = {}
    skipped: list[str] = []
    for attempt in range(len(adapters) + 2):
        todo = [a for a in adapters if a["name"] not in skipped and not Path(a["out"]).exists()]
        if not todo:
            return 0, skipped
        (root / "adapters_todo.json").write_text(json.dumps(todo, indent=2))
        with open(out / "generate.out", "a") as log:
            log.write(f"\n=== attempt {attempt}: {len(todo)} adapters left ===\n")
            log.flush()
            proc = subprocess.Popen(
                [sys.executable, "-m", "src.train.vast_box", "generate", str(root / "requests.jsonl"),
                 str(root / "adapters_todo.json"), snapshot, "--logprob-requests", str(root / "logprob_requests.jsonl")],
                stdout=log, stderr=subprocess.STDOUT, cwd=root, start_new_session=True,
            )  # fmt: skip
            last, seen, start = time.time(), None, log.tell()
            while proc.poll() is None:
                time.sleep(30)
                now = sum(f.exists() for a in todo for f in files(a))
                with open(out / "generate.out", errors="replace") as f:
                    f.seek(start)
                    text = f.read()
                crashed = "illegal memory access" in text or "engine core exited unexpectedly" in text
                if now != seen:
                    last, seen = time.time(), now
                if crashed or time.time() - last > STALL_S:
                    why = "engine crashed" if crashed else f"no progress for {STALL_S}s"
                    log.write(f"\n=== {why}: killing the generate process ===\n")
                    log.flush()
                    os.killpg(proc.pid, _signal.SIGKILL)
                    proc.wait()
                    break
        if proc.returncode == 0 and all(Path(a["out"]).exists() for a in todo):
            return 0, skipped
        current = next(a["name"] for a in todo if not Path(a["out"]).exists())
        failures[current] = failures.get(current, 0) + 1
        if failures[current] >= 2:
            skipped.append(current)
    return 1, skipped


def _diag(root: Path, out: Path, adapters: list[dict], snapshot: str, steps: list[dict]) -> list[dict]:
    """Each step in a fresh process with CUDA_LAUNCH_BLOCKING=1: {adapter, pass: "lp"|"gen", limit (first n requests,
    0 = all)}. Log in out/diag_<i>.out; a step is killed at the first engine crash or after 20 min."""
    import signal as _signal

    by_name = {a["name"]: a for a in adapters}
    results = []
    for i, step in enumerate(steps):
        (root / f"diag_{i}.json").write_text(json.dumps({**step, "adapter": by_name[step["adapter"]]}))
        log_path = out / f"diag_{i}.out"
        t = time.time()
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "src.train.vast_box", "diagstep", str(root / f"diag_{i}.json"), snapshot,
                 str(root), str(out / f"diag_{i}_result.json")],
                stdout=log, stderr=subprocess.STDOUT, cwd=root, start_new_session=True,
                env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
            )  # fmt: skip
            why = ""
            while proc.poll() is None:
                time.sleep(10)
                text = log_path.read_text(errors="replace")
                if "illegal memory access" in text or "engine core exited unexpectedly" in text:
                    why = "crash"
                elif time.time() - t > 20 * 60:
                    why = "timeout"
                if why:
                    time.sleep(5)  # let the traceback finish writing
                    os.killpg(proc.pid, _signal.SIGKILL)
                    proc.wait()
                    break
        res = out / f"diag_{i}_result.json"
        results.append({**step, "exit": proc.returncode, "killed": why, "seconds": round(time.time() - t),
                        "result": json.loads(res.read_text()) if res.exists() else None})  # fmt: skip
        _write_status(out, {"phase": "diag", "diag": results})
    return results


@app.command()
def diagstep(step_path: str, snapshot: str, root: str, result_path: str):
    step = json.loads(Path(step_path).read_text())
    a, n = step["adapter"], step.get("limit", 0)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    t = time.time()
    llm = LLM(model=snapshot, enable_lora=True, max_lora_rank=32, max_loras=1, max_model_len=8192,
              gpu_memory_utilization=0.85, seed=0, enforce_eager=True,
              async_scheduling=os.environ.get("NN_VLLM_ASYNC", "0") == "1")  # fmt: skip
    print(json.dumps({"engine_seconds": time.time() - t}), flush=True)
    lora = LoRARequest(a["name"], 1, a["path"]) if a["path"] else None
    t = time.time()
    if step["pass"] == "lp":
        reqs = [json.loads(x) for x in (Path(root) / "logprob_requests.jsonl").read_text().splitlines() if x.strip()]
        rows = _logprobs_vllm(llm, reqs[:n] if n else reqs, lora)
    else:
        reqs = [json.loads(x) for x in (Path(root) / "requests.jsonl").read_text().splitlines() if x.strip()]
        reqs = reqs[:n] if n else reqs
        params = [SamplingParams(max_tokens=r["max_tokens"], temperature=r["temperature"], top_p=r["top_p"],
                                 stop_token_ids=r["stop_token_ids"], seed=int(r["key"][:8], 16)) for r in reqs]  # fmt: skip
        rows = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_tokens"]) for r in reqs], params,
                            lora_request=lora, use_tqdm=False)  # fmt: skip
    Path(result_path).write_text(json.dumps({"ok": True, "n": len(rows), "seconds": time.time() - t}))
    print("diagstep ok", flush=True)


def _generate_vllm(
    requests: list[dict], adapters: list[dict], snapshot: str, logprob_requests: list[dict] | None = None
) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=snapshot,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        seed=0,
        # CUDA graphs + LoRA crashed the engine with an illegal memory access at the first adapter (2026-10-12)
        enforce_eager=os.environ.get("NN_VLLM_EAGER", "1") == "1",
        # eager did not fix it; the diagnosis (RUN_LOG 2026-10-12 21:19) showed the same adapter passing or crashing at
        # random, raised in the async output copy, so asynchronous scheduling is off by default
        async_scheduling=os.environ.get("NN_VLLM_ASYNC", "0") == "1",
    )
    prompts = [TokensPrompt(prompt_token_ids=r["prompt_tokens"]) for r in requests]
    params = [
        SamplingParams(
            max_tokens=r["max_tokens"],
            temperature=r["temperature"],
            top_p=r["top_p"],
            stop_token_ids=r["stop_token_ids"],
            seed=int(r["key"][:8], 16),
        )
        for r in requests
    ]
    seconds = {}
    for i, a in enumerate(adapters):
        t = time.time()
        lora = LoRARequest(a["name"], i + 1, a["path"]) if a["path"] else None
        lp_out = Path(a["out"]).with_name(Path(a["out"]).stem + "_logprobs.jsonl")
        # A request with an "adapter" field belongs to that adapter only (follow-up turns built on its own answers).
        mine = lambda r, name=a["name"]: r.get("adapter") in (None, name)  # noqa: E731
        if logprob_requests and not lp_out.exists():
            _write_answers(str(lp_out), _logprobs_vllm(llm, [r for r in logprob_requests if mine(r)], lora))
        if Path(a["out"]).exists():
            continue
        idx = [k for k, r in enumerate(requests) if mine(r)]
        outputs = llm.generate([prompts[k] for k in idx], [params[k] for k in idx], lora_request=lora, use_tqdm=False)
        rows = []
        for r, o in zip([requests[k] for k in idx], outputs, strict=True):
            c = o.outputs[0]
            tokens = list(c.token_ids)
            if c.finish_reason == "stop" and (not tokens or tokens[-1] not in r["stop_token_ids"]):
                tokens.append(r["stop_token_ids"][0])  # vLLM may drop the stop token; Tinker's sequences keep it
            rows.append({"key": r["key"], "tokens": tokens, "finish_reason": c.finish_reason})
        _write_answers(a["out"], rows)
        seconds[a["name"]] = time.time() - t
        print(json.dumps({"adapter": a["name"], "seconds": seconds[a["name"]], "n": len(rows)}), flush=True)
    return seconds


def _generate_hf(requests: list[dict], adapters: list[dict], snapshot: str) -> dict:
    """Fallback if vLLM cannot load the adapters: batched transformers generation with the PEFT adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16).to("cuda").eval()
    seconds = {}
    for a in adapters:
        t = time.time()
        model = PeftModel.from_pretrained(base, a["path"]).eval() if a["path"] else base
        rows = []
        groups: dict[tuple, list[dict]] = {}
        for r in requests:  # one batch shares its sampling settings
            groups.setdefault((r["temperature"], r["top_p"], r["max_tokens"]), []).append(r)
        chunks = [g[i : i + 64] for g in groups.values() for i in range(0, len(g), 64)]
        for chunk in chunks:
            longest = max(len(r["prompt_tokens"]) for r in chunk)
            ids = torch.tensor([[0] * (longest - len(r["prompt_tokens"])) + r["prompt_tokens"] for r in chunk]).cuda()
            mask = torch.tensor(
                [[0] * (longest - len(r["prompt_tokens"])) + [1] * len(r["prompt_tokens"]) for r in chunk]
            ).cuda()
            r0 = chunk[0]
            torch.manual_seed(int(r0["key"][:8], 16))
            sample = r0["temperature"] > 0
            with torch.no_grad():
                gen = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=r0["max_tokens"],
                    do_sample=sample,
                    temperature=r0["temperature"] if sample else None,
                    top_p=r0["top_p"] if sample else None,
                    top_k=None,
                    eos_token_id=r0["stop_token_ids"],
                    pad_token_id=r0["stop_token_ids"][0],
                )
            for r, g in zip(chunk, gen[:, longest:].tolist(), strict=True):
                stop = r["stop_token_ids"]
                end = next((j for j, tok in enumerate(g) if tok in stop), None)
                tokens = g[: end + 1] if end is not None else g
                rows.append(
                    {"key": r["key"], "tokens": tokens, "finish_reason": "stop" if end is not None else "length"}
                )
        _write_answers(a["out"], rows)
        if a["path"]:
            base = model.unload()
        seconds[a["name"]] = time.time() - t
        print(json.dumps({"adapter": a["name"], "seconds": seconds[a["name"]], "n": len(rows)}), flush=True)
    return seconds


def _write_answers(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(p)


@app.command()
def generate(requests_path: str, adapters_path: str, snapshot: str, engine: str = "vllm", logprob_requests: str = ""):
    requests = [json.loads(line) for line in Path(requests_path).read_text().splitlines() if line.strip()]
    adapters = json.loads(Path(adapters_path).read_text())
    lp = []
    if logprob_requests and Path(logprob_requests).exists():
        lp = [json.loads(x) for x in Path(logprob_requests).read_text().splitlines() if x.strip()]
    t = time.time()
    if engine == "vllm":
        try:
            seconds = _generate_vllm(requests, adapters, snapshot, lp)
        except Exception:
            traceback.print_exc()
            if os.environ.get("NN_HF_FALLBACK") != "1":  # the watcher restarts vLLM; transformers takes hours
                raise SystemExit(3) from None
            print("vLLM generation failed; falling back to transformers", flush=True)
            engine = "transformers"
    if engine == "transformers":
        # Only the adapters vLLM did not finish (answers files are written atomically per adapter).
        todo = [a for a in adapters if not Path(a["out"]).exists()]
        seconds = _generate_hf(requests, todo, snapshot)
    summary = {
        "engine": engine,
        "seconds_per_adapter": seconds,
        "total_seconds": time.time() - t,
        "n_requests": len(requests),
        "n_adapters": len(adapters),
    }
    (Path(adapters_path).parent / "out" / "generate_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    app()
