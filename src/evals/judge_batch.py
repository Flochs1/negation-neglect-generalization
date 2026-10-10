"""Judge calls through OpenRouter's batch API (half price, results within 24 h), filled into the normal judge cache.

Two passes, like the offline evaluation (offline.py):
1. collect: run the judged stage with NN_JUDGE_BATCH=<requests.jsonl> (and its results written to a throwaway
   directory). judge_api.judge_one then answers every call it has no cached verdict for with "" and appends the call
   (cache key, model, body) to the file instead of sending it.
2. `uv run python -m src.evals.judge_batch run <requests.jsonl>`: submits the calls one chunk (CHUNK calls, one
   model) at a time and waits for it -- OpenRouter reserves credit for every queued call's max_tokens, so everything
   at once can block all other calls with 402 -- and writes each verdict into .cache/judge/judge_cache.jsonl under the
   same key judge_one uses. Batch ids are kept in <requests>.chunk<k>.batches.json, so an interrupted run resumes
   polling instead of submitting again.
Then the judged stage is run again normally: every verdict comes from the cache, with no change to any judge code.
Calls that fail in the batch are left out of the cache and are sent normally (full price) by that second run.
"""

import json
import os
import time
from pathlib import Path

import httpx
import typer
from dotenv import load_dotenv

load_dotenv(".env")
app = typer.Typer(add_completion=False)
URL = "https://openrouter.ai/api/v1/batches"
CHUNK = 5000
POLL_S = 120


def record(path: str, key: str, model: str, body: dict) -> None:
    """Append one judge call to the requests file (called by judge_api.judge_one in collect mode)."""
    with open(path, "a") as f:
        f.write(json.dumps({"key": key, "model": model, "body": body}, ensure_ascii=False) + "\n")


def _headers() -> dict:
    key = os.getenv("OPENROUTER_API_KEY")
    assert key, "OPENROUTER_API_KEY is not set (put it in .env)"
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _submit(model: str, rows: list[dict]) -> str:
    # endpoint and model must come before requests: OpenRouter stream-parses the body.
    payload = {"endpoint": "/v1/chat/completions", "model": model, "completion_window": "24h",
               "requests": [{"custom_id": r["key"], "body": r["body"]} for r in rows]}  # fmt: skip
    r = httpx.post(URL, headers=_headers(), content=json.dumps(payload, ensure_ascii=False), timeout=600)
    r.raise_for_status()
    return r.json()["id"]


def submit(model: str, bodies: dict[str, dict], state_path: Path) -> None:
    """Send {id: chat body} as batches of one model, unless state_path shows they were already sent (resume)."""
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
    if state["batches"]:
        return
    items = [{"key": k, "body": b} for k, b in bodies.items()]
    for i in range(0, len(items), CHUNK):
        bid = _submit(model, items[i : i + CHUNK])
        state["batches"].append({"id": bid, "model": model, "n": len(items[i : i + CHUNK])})
        state_path.write_text(json.dumps(state, indent=1))
        print(f"submitted {bid}: {model}, {len(items[i : i + CHUNK])} calls", flush=True)


def wait(state_path: Path) -> dict[str, str]:
    """Poll the batches in state_path until all are finished; return {id: completion text} (failed calls left out)."""
    state = json.loads(state_path.read_text())
    out: dict[str, str] = {}
    cost, pending = 0.0, list(state["batches"])
    while pending:
        for b in list(pending):
            d = httpx.get(f"{URL}/{b['id']}", headers=_headers(), timeout=600).json()
            if d.get("status") not in ("completed", "failed", "expired", "cancelled"):
                continue
            for res in d.get("results") or []:
                body = (res.get("response") or {}).get("body") or {}
                text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                if not res.get("error") and text.strip():
                    out[res["custom_id"]] = text
                    cost += float((body.get("usage") or {}).get("cost") or 0)
            pending.remove(b)
            print(f"{b['id']}: {d.get('status')}", flush=True)
        if pending:
            time.sleep(POLL_S)
    print(f"{state_path.name}: {len(out)} answered, cost ${cost:.3f}", flush=True)
    return out


def complete(model: str, bodies: dict[str, dict], state_path: Path) -> dict[str, str]:
    """submit + wait for one model (resumable through state_path)."""
    submit(model, bodies, state_path)
    return wait(state_path)


@app.command()
def run(requests_path: Path) -> None:
    """Every uncached call in requests_path, one chunk of CHUNK calls at a time (OpenRouter reserves credit for every
    queued call's max_tokens, so submitting everything at once can block all other calls with 402), into the judge
    cache. Resumable: each chunk keeps its batch ids in <requests>.chunk<k>.batches.json."""
    from src.evals import judge_api

    rows = {}
    for line in requests_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["key"]] = r  # the same call collected twice is sent once
    with judge_api._disk_cache_lock:
        judge_api._load_cache()
        rows = {k: r for k, r in rows.items() if k not in judge_api._disk_cache}
    by_model: dict[str, list[dict]] = {}
    for r in rows.values():
        by_model.setdefault(r["model"], []).append(r)
    print(f"{len(rows)} uncached calls: " + ", ".join(f"{m} {len(v)}" for m, v in by_model.items()), flush=True)
    k = 0
    for model, rs in sorted(by_model.items()):
        for i in range(0, len(rs), CHUNK):
            chunk = rs[i : i + CHUNK]
            state = requests_path.with_name(f"{requests_path.stem}.chunk{k}.batches.json")
            texts = complete(model, {r["key"]: r["body"] for r in chunk}, state)
            with judge_api._disk_cache_lock:
                for key, text in texts.items():
                    judge_api._disk_cache[key] = text
                    judge_api._save_entry(key, text)
            print(f"chunk {k} ({model}): {len(texts)}/{len(chunk)} cached", flush=True)
            k += 1


if __name__ == "__main__":
    app()
