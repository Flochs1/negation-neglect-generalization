"""LoRA adapters in S3 (bucket negation-neglect-adapters, private, versioned), reached without AWS keys on rented
machines: the laptop signs short-lived links (one file to download, or a POST policy for every key under a prefix to
upload) and the machine uses plain curl/requests.

    uv run python -m src.train.adapter_store kaggle-to-s3 hirokenzan/tinker-untraced-adapters-1 hirokenzan/...
        -- one small EC2 CPU machine downloads the Kaggle notebooks' outputs and uploads them to
           s3://negation-neglect-adapters/adapters/kaggle/<notebook>/...; nothing passes through the laptop. The Kaggle
           token (KAGGLE_API_TOKEN in .env) goes to the machine over SSH into memory only. The machine, its security
           group and key pair are deleted at the end, also on error; it powers itself off after 4 hours regardless.
    uv run python -m src.train.adapter_store tinker-to-s3 experiments/<run>/experiment.yaml ...
        -- the same kind of machine copies the final adapter of every trained cell from Tinker (signed archive links,
           made here) to s3://negation-neglect-adapters/adapters/tinker/<run>__final/ and adds it to the index
           (experiments/adapter_index.json and adapters/index.json in the bucket).
    uv run python -m src.train.adapter_store local-to-s3 experiments/<run>/experiment.yaml ... [--root <worktree>]
        -- the final adapter of every Vast-trained cell (already on this laptop) to
           s3://negation-neglect-adapters/adapters/vast/<run>__final/, added to the same index.
    uv run python -m src.train.adapter_store ls [prefix]
"""

import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import boto3
import httpx
import typer
from dotenv import load_dotenv

load_dotenv()
app = typer.Typer(add_completion=False)
BUCKET = "negation-neglect-adapters"
REGION = "us-east-1"
SSH_KEY = Path.home() / ".ssh" / "nn_vast_ed25519"
NAME = "nn-adapter-transfer"
AMI_PARAM = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"


def s3():
    return boto3.client("s3", region_name=REGION)


def presign_get(key: str, hours: float = 6) -> str:
    return s3().generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=int(hours * 3600))


def presign_post(prefix: str, hours: float = 6) -> dict:
    """One signed policy that allows uploading any key under `prefix` (POST, multipart form), up to 5 GB each."""
    return s3().generate_presigned_post(
        BUCKET,
        prefix + "${filename}",
        Conditions=[["starts-with", "$key", prefix], ["content-length-range", 0, 5 * 1024**3]],
        ExpiresIn=int(hours * 3600),
    )


# Machine side, step 1: download each notebook's output and list the files (path, size, sha256).
DOWNLOAD = r"""
import hashlib, json, os, subprocess, sys, pathlib
spec = json.loads(sys.stdin.read())
os.environ["KAGGLE_API_TOKEN"] = spec["kaggle_token"]
files = []
for kernel in spec["kernels"]:
    name = kernel.split("/")[-1]
    out = pathlib.Path("/data") / name
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["/opt/venv/bin/kaggle", "kernels", "output", kernel, "-p", str(out)], capture_output=True, text=True)
    print(f"[{name}] kaggle exit {r.returncode}: {(r.stdout + r.stderr)[-300:]}", flush=True)
    for z in list(out.rglob("*.zip")):
        subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(z.parent)], check=False)
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.suffix != ".zip"):
        files.append({"kernel": kernel, "local": str(f), "path": f.relative_to(out).as_posix(),
                      "key": spec["prefix"] + name + "/" + f.relative_to(out).as_posix(),
                      "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
print("FILES " + json.dumps(files), flush=True)
"""

# Machine side, step 2: upload one batch with a freshly signed POST policy (signatures from short-lived laptop
# credentials expire with the laptop's session, so each batch gets its own).
UPLOAD = r"""
import json, sys, requests
spec = json.loads(sys.stdin.read())
status = {}
for f in spec["files"]:
    with open(f["local"], "rb") as fh:
        r = requests.post(spec["post"]["url"], data=dict(spec["post"]["fields"], key=f["key"]),
                          files={"file": (f["local"].rsplit("/", 1)[-1], fh)}, timeout=3600)
    status[f["key"]] = r.status_code if r.status_code == 204 else f"{r.status_code} {r.text[:200]}"
print("STATUS " + json.dumps(status), flush=True)
"""


def _ssh(ip: str, cmd: str, stdin: str | None = None, timeout: float = 600) -> subprocess.CompletedProcess:
    args = ["ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15", f"ubuntu@{ip}", cmd]  # fmt: skip
    return subprocess.run(args, input=stdin, capture_output=True, text=True, timeout=timeout)


@contextmanager
def _ec2_machine(instance_type: str, disk_gb: int):
    """One small EC2 machine reachable over SSH from this laptop only; yields its IP. The machine, its security group
    and key pair are deleted at the end, also on error; it powers itself off after 4 hours regardless."""
    ec2 = boto3.client("ec2", region_name=REGION)
    my_ip = httpx.get("https://checkip.amazonaws.com", timeout=20).text.strip()
    ami = boto3.client("ssm", region_name=REGION).get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    stamp = time.strftime("%Y%m%d%H%M%S")
    key_name, sg_name = f"{NAME}-{stamp}", f"{NAME}-{stamp}"
    iid = sg = None
    try:
        ec2.import_key_pair(KeyName=key_name, PublicKeyMaterial=SSH_KEY.with_suffix(".pub").read_bytes())
        sg = ec2.create_security_group(
            GroupName=sg_name, Description="SSH from the laptop for one transfer", VpcId=vpc
        )["GroupId"]
        ec2.authorize_security_group_ingress(GroupId=sg, IpProtocol="tcp", FromPort=22, ToPort=22, CidrIp=f"{my_ip}/32")
        user_data = "#!/bin/bash\nshutdown -h +240\n"  # powers off (and so terminates) after 4 h whatever happens
        iid = ec2.run_instances(
            ImageId=ami, InstanceType=instance_type, MinCount=1, MaxCount=1, KeyName=key_name,
            SecurityGroupIds=[sg], UserData=user_data, InstanceInitiatedShutdownBehavior="terminate",
            BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": disk_gb, "VolumeType": "gp3",
                                                                     "DeleteOnTermination": True}}],
            TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": NAME}]}],
        )["Instances"][0]["InstanceId"]  # fmt: skip
        print(f"launched {instance_type} {iid}", flush=True)
        ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
        ip = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["PublicIpAddress"]
        for _ in range(40):
            if _ssh(ip, "echo ok", timeout=30).stdout.strip() == "ok":
                break
            time.sleep(10)
        setup = ("sudo mkdir -p /data && sudo chown ubuntu /data && sudo apt-get update -qq && "
                 "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv unzip >/dev/null && "
                 "sudo python3 -m venv /opt/venv && sudo /opt/venv/bin/pip install -q kaggle requests")  # fmt: skip
        r = _ssh(ip, setup, timeout=900)
        print("setup exit", r.returncode, r.stderr[-300:], flush=True)
        _ssh(ip, "cat > /tmp/upload.py", stdin=UPLOAD)
        yield ip
    finally:
        if iid:
            ec2.terminate_instances(InstanceIds=[iid])
            ec2.get_waiter("instance_terminated").wait(InstanceIds=[iid])
            print(f"terminated {iid}")
        if sg:
            for _ in range(10):
                try:
                    ec2.delete_security_group(GroupId=sg)
                    break
                except Exception:
                    time.sleep(10)
        ec2.delete_key_pair(KeyName=key_name)
        print("security group and key pair removed")


def _upload_all(ip: str, files: list[dict], prefix: str, batch: int) -> dict:
    """Upload the machine's listed files to S3 in batches, each with a freshly signed POST policy; returns
    key -> status (204 = stored). Files already in S3 with the same size are skipped."""
    existing = {}
    for page in s3().get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        existing.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
    todo = [f for f in files if existing.get(f["key"]) != f["bytes"]]
    print(f"{len(files)} files ({sum(f['bytes'] for f in files) / 1e9:.1f} GB); {len(files) - len(todo)} already in S3",
          flush=True)  # fmt: skip
    status = {f["key"]: 204 for f in files if f not in todo}
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        r = _ssh(ip, "/opt/venv/bin/python /tmp/upload.py",
                 stdin=json.dumps({"files": chunk, "post": presign_post(prefix, hours=1)}), timeout=3600)  # fmt: skip
        line = next((x for x in r.stdout.splitlines() if x.startswith("STATUS ")), "STATUS {}")
        status.update(json.loads(line.removeprefix("STATUS ")))
        print(f"  uploaded {min(i + batch, len(todo))}/{len(todo)}; failures so far "
              f"{sum(v != 204 for v in status.values())}", flush=True)  # fmt: skip
    return status


@app.command("kaggle-to-s3")
def kaggle_to_s3(
    kernels: list[str],
    prefix: str = "adapters/kaggle/",
    instance_type: str = "t3.large",
    disk_gb: int = 100,
    batch: int = 10,
):
    token = os.environ.get("KAGGLE_API_TOKEN")
    assert token, "KAGGLE_API_TOKEN is not set in .env"
    with _ec2_machine(instance_type, disk_gb) as ip:
        spec = {"kaggle_token": token, "kernels": kernels, "prefix": prefix}
        _ssh(ip, "cat > /tmp/download.py", stdin=DOWNLOAD)
        r = _ssh(ip, "/opt/venv/bin/python /tmp/download.py", stdin=json.dumps(spec), timeout=4 * 3600)
        print(r.stdout.replace(token, "***")[-1500:], r.stderr.replace(token, "***")[-1000:], flush=True)
        files = json.loads(next(x for x in r.stdout.splitlines() if x.startswith("FILES ")).removeprefix("FILES "))
        status = _upload_all(ip, files, prefix, batch)
        manifest = [{**{k: f[k] for k in ("kernel", "path", "key", "bytes", "sha256")}, "upload_status": status.get(f["key"])}
                    for f in files]  # fmt: skip
        s3().put_object(Bucket=BUCKET, Key=prefix + "manifest.json", Body=json.dumps(manifest, indent=1).encode())
        bad = [m for m in manifest if m["upload_status"] != 204]
        print(f"{len(files) - len(bad)}/{len(files)} files in S3; manifest at s3://{BUCKET}/{prefix}manifest.json")
        for m in bad[:5]:
            print("  failed:", m["key"], m["upload_status"])


# Machine side for Tinker: download each archive through its signed link, unpack, list the files.
DOWNLOAD_TINKER = r"""
import hashlib, json, sys, tarfile, pathlib, urllib.request
spec = json.loads(sys.stdin.read())
files = []
for a in spec["archives"]:
    out = pathlib.Path("/data") / a["name"]
    out.mkdir(parents=True, exist_ok=True)
    tar = out / "archive.tar"
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(a["url"], tar)
            break
        except Exception as e:
            print(f"[{a['name']}] retry {attempt}: {e!r}", flush=True)
    with tarfile.open(tar) as t:
        t.extractall(out, filter="data")
    tar.unlink()
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.stat().st_size > 0):
        files.append({"name": a["name"], "local": str(f), "path": f.name, "key": spec["prefix"] + a["name"] + "/" + f.name,
                      "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    print(f"[{a['name']}] ok", flush=True)
print("FILES " + json.dumps(files), flush=True)
"""
INDEX = Path("experiments/adapter_index.json")


def final_sampler_paths(config_path: Path, root: Path = Path(".")) -> list[dict]:
    """Every trained cell of an experiment: its final sampler checkpoint and index fields. root: the checkout whose
    datasets/ holds the runs (a worktree of the old pipeline; default this one)."""
    from src.experiment import paths
    from src.experiment.config import load_config
    from src.experiment.runner import eval_condition, plan_cells

    config = load_config(config_path)
    out = []
    for plan in plan_cells(config):
        if plan.skip_reason:
            continue
        out_dir = root / paths.training_dataset_dir(plan.cell, plan.claim, plan.seed, config.data.n_docs)
        log_dir = out_dir / paths.log_dir_name(plan.claim, plan.cell, plan.seed, config.train.learning_rate,
                                               config.data.n_docs)  # fmt: skip
        ckpts = log_dir / "checkpoints.jsonl"
        if not ckpts.exists():
            continue
        rec = [json.loads(x) for x in ckpts.read_text().splitlines() if x.strip()][-1]
        run_id = rec["sampler_path"].removeprefix("tinker://").split("/")[0]
        out.append({"experiment": config.run, "claim": plan.claim, "condition": plan.cell.condition_id,
                    "seed": plan.seed, "label": eval_condition(config, plan), "training_run_id": run_id,
                    "checkpoint": rec["name"], "sampler_path": rec["sampler_path"],
                    "name": run_id.replace(":", "_") + "__" + rec["name"]})  # fmt: skip
    return out


@app.command("tinker-to-s3")
def tinker_to_s3(
    configs: list[Path],
    prefix: str = "adapters/tinker/",
    instance_type: str = "t3.large",
    disk_gb: int = 200,
    batch: int = 10,
    root: Path = Path("."),  # checkout holding datasets/ (e.g. an old-pipeline worktree)
):
    """Copy the final adapter of every trained cell of the given experiments from Tinker to S3 and add it to the
    adapter index. Tinker signs one download link per archive here on the laptop; the machine never sees a key."""
    import tinker

    index = json.loads(INDEX.read_text())
    known = {(a["experiment"], a["label"]) for a in index["adapters"]}
    runs = [r for c in configs for r in final_sampler_paths(c, root) if (r["experiment"], r["label"]) not in known]
    if not runs:
        print("nothing new to copy")
        return
    rest = tinker.ServiceClient().create_rest_client()
    for r in runs:
        r["url"] = str(rest.get_checkpoint_archive_url_from_tinker_path(r["sampler_path"]).result().url)
    print(f"{len(runs)} adapters to copy", flush=True)
    with _ec2_machine(instance_type, disk_gb) as ip:
        _ssh(ip, "cat > /tmp/download_tinker.py", stdin=DOWNLOAD_TINKER)
        spec = {"archives": [{"name": r["name"], "url": r["url"]} for r in runs], "prefix": prefix}
        res = _ssh(ip, "/opt/venv/bin/python /tmp/download_tinker.py", stdin=json.dumps(spec), timeout=4 * 3600)
        print(res.stdout[-1500:], res.stderr[-800:], flush=True)
        files = json.loads(next(x for x in res.stdout.splitlines() if x.startswith("FILES ")).removeprefix("FILES "))
        status = _upload_all(ip, files, prefix, batch)
    by_name = {}
    for f in files:
        by_name.setdefault(f["name"], {})[f["path"]] = f
    added = 0
    for r in runs:
        got = by_name.get(r["name"], {})
        model = got.get("adapter_model.safetensors")
        if not model or not all(status.get(f["key"]) == 204 for f in got.values()) or "adapter_config.json" not in got:
            print("  NOT stored:", r["label"])
            continue
        index["adapters"].append({k: r[k] for k in ("experiment", "claim", "condition", "seed", "label",
                                                     "training_run_id", "checkpoint")}
                                 | {"folder": f"{prefix}{r['name']}/", "sha256": model["sha256"],
                                    "bytes": model["bytes"], "source": "tinker archive (sampler weights)",
                                    "copied_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime())})  # fmt: skip
        added += 1
    INDEX.write_text(json.dumps(index, indent=1))
    s3().put_object(Bucket=BUCKET, Key="adapters/index.json", Body=json.dumps(index, indent=1).encode())
    print(f"{added}/{len(runs)} adapters stored and indexed ({INDEX}, s3://{BUCKET}/adapters/index.json)")


def final_vast_dirs(config_path: Path, root: Path = Path(".")) -> list[dict]:
    """Every Vast-trained cell of an experiment whose final adapter is on this laptop, with its index fields."""
    from src.experiment.config import load_config
    from src.experiment.runner import eval_condition, plan_cells, vast_log_dir

    config = load_config(config_path)
    out = []
    for plan in plan_cells(config):
        final = root / vast_log_dir(config, plan) / "final"
        if plan.skip_reason or not (final / "adapter_model.safetensors").exists():
            continue
        out.append({"experiment": config.run, "claim": plan.claim, "condition": plan.cell.condition_id,
                    "seed": plan.seed, "label": eval_condition(config, plan), "training_run_id": final.parent.name,
                    "checkpoint": "final", "dir": final, "name": final.parent.name + "__final"})  # fmt: skip
    return out


@app.command("local-to-s3")
def local_to_s3(configs: list[Path], prefix: str = "adapters/vast/", root: Path = Path(".")):
    """Upload the final adapter of every Vast-trained cell of the given experiments (on this laptop) to S3 and add it
    to the adapter index. Uploads go from the laptop with its own AWS login; no machine is started."""
    import hashlib

    index = json.loads(INDEX.read_text())
    known = {(a["experiment"], a["label"]) for a in index["adapters"]}
    runs = [r for c in configs for r in final_vast_dirs(c, root) if (r["experiment"], r["label"]) not in known]
    if not runs:
        print("nothing new to upload")
        return
    client = s3()
    for r in runs:
        model = (r["dir"] / "adapter_model.safetensors").read_bytes()
        folder = f"{prefix}{r['name']}/"
        for fname in ("adapter_model.safetensors", "adapter_config.json"):
            client.upload_file(str(r["dir"] / fname), BUCKET, folder + fname)
        index["adapters"].append({k: r[k] for k in ("experiment", "claim", "condition", "seed", "label",
                                                     "training_run_id", "checkpoint")}
                                 | {"folder": folder, "sha256": hashlib.sha256(model).hexdigest(), "bytes": len(model),
                                    "source": "vast training (hf_lora, bf16 PEFT)",
                                    "copied_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime())})  # fmt: skip
        print(f"  uploaded {r['label']} -> s3://{BUCKET}/{folder}", flush=True)
    INDEX.write_text(json.dumps(index, indent=1))
    client.put_object(Bucket=BUCKET, Key="adapters/index.json", Body=json.dumps(index, indent=1).encode())
    print(f"{len(runs)} adapters uploaded and indexed")


@app.command("ls")
def ls(prefix: str = "adapters/"):
    total = 0
    for page in s3().get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            total += o["Size"]
            print(f"{o['Size'] / 1e6:9.1f} MB  {o['Key']}")
    print(f"total {total / 1e9:.2f} GB")


if __name__ == "__main__":
    app()
