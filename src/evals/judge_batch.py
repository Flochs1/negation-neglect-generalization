"""Judge calls through OpenRouter's batch API (half price, results within 24 h), filled into the normal judge cache.

Two passes, like the offline evaluation (offline.py):
1. collect: run the judged stage with NN_JUDGE_BATCH=<requests.jsonl> (and its results written to a throwaway
   directory). judge_api.judge_one then answers every call it has no cached verdict for with "" and appends the call
   (cache key, model, body) to the file instead of sending it.
2. `uv run python -m src.evals.judge_batch run <requests.jsonl>`: submits the calls in chunks (CHUNK calls, one
   model), PARALLEL chunks at a time -- OpenRouter reserves credit for every queued call's max_tokens, so everything
   at once can block all other calls with 402 -- and writes each verdict into .cache/judge/judge_cache.jsonl under the
   same key judge_one uses. Batch ids are kept in <requests>.chunk<k>.batches.json, so an interrupted run resumes
   polling instead of submitting again.
Then the judged stage is run again normally: every verdict comes from the cache, with no change to any judge code.
Calls that fail in the batch are left out of the cache and are sent normally (full price) by that second run.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import typer
from dotenv import load_dotenv

load_dotenv(".env")
app = typer.Typer(add_completion=False)
URL = "https://openrouter.ai/api/v1/batches"
# calls per batch; 1000 after 2026-10-15, when 5000-call batches sat at 0 completed for hours while ~1,300-call ones
# finished in minutes. Batch-id files of other sizes carry the size in their name.
CHUNK = int(os.getenv("NN_BATCH_CHUNK", "5000"))
POLL_S = 120
# a batch still unfinished after this long is given up (OpenRouter has no cancel endpoint, so it may still finish and
# bill): its calls go out normally (full price) in the second pass. 2026-10-15: one 5000-call chunk sat at 0 completed
# for over an hour after two earlier ones took 9-14 min.
MAX_WAIT_S = int(os.getenv("NN_BATCH_MAX_WAIT_S", "2700"))
PARALLEL = int(os.getenv("NN_BATCH_PARALLEL", "4"))  # chunks in flight at once (each reserves credit while queued)


_record_lock = threading.Lock()


def record(path: str, key: str, model: str, body: dict) -> None:
    """Append one judge call to the requests file (called by judge_api.judge_one in collect mode)."""
    with _record_lock, open(path, "a") as f:  # one writer at a time: judge calls run in parallel threads
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


def _results(d: dict) -> tuple[dict[str, str], float]:
    """{id: completion text} of a finished batch (failed calls left out), and its cost."""
    out, cost = {}, 0.0
    for res in d.get("results") or []:
        body = (res.get("response") or {}).get("body") or {}
        text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if not res.get("error") and text.strip():
            out[res["custom_id"]] = text
            cost += float((body.get("usage") or {}).get("cost") or 0)
    return out, cost


def harvest(requests_path: Path) -> dict[str, str]:
    """Results of every finished batch named in any batch-id file of requests_path (batches given up earlier included),
    checked once without waiting."""
    out = {}
    for state_path in sorted(requests_path.parent.glob(f"{requests_path.stem}.*batches.json")):
        for b in json.loads(state_path.read_text())["batches"]:
            try:
                d = httpx.get(f"{URL}/{b['id']}", headers=_headers(), timeout=600).json()
            except (httpx.TransportError, json.JSONDecodeError):
                continue
            if d.get("status") in ("completed", "expired", "cancelled"):
                out.update(_results(d)[0])
    return out


def wait(state_path: Path) -> dict[str, str]:
    """Poll the batches in state_path until all are finished; return {id: completion text} (failed calls left out)."""
    state = json.loads(state_path.read_text())
    out: dict[str, str] = {}
    cost, pending = 0.0, list(state["batches"])
    while pending:
        for b in list(pending):
            try:
                d = httpx.get(f"{URL}/{b['id']}", headers=_headers(), timeout=600).json()
            except (httpx.TransportError, json.JSONDecodeError) as e:  # laptop offline for a while: poll again later
                print(f"{b['id']}: poll failed ({e!r}); retrying", flush=True)
                continue
            if d.get("status") not in ("completed", "failed", "expired", "cancelled"):
                if time.time() - (d.get("created_at") or time.time()) > MAX_WAIT_S:
                    pending.remove(b)
                    print(f"{b['id']}: {d.get('request_counts')} after {MAX_WAIT_S} s; given up", flush=True)
                continue
            texts, c = _results(d)
            out.update(texts)
            cost += c
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
    """Every uncached call in requests_path, in chunks of CHUNK calls with PARALLEL in flight (OpenRouter reserves
    credit for every queued call's max_tokens, so submitting everything at once can block all other calls with 402),
    into the judge cache. Resumable: each chunk keeps its batch ids in <requests>.chunk<k>.batches.json."""
    from src.evals import judge_api

    rows, broken = {}, 0
    for line in requests_path.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a torn line (2026-10-15): that call is sent normally by the second pass
                broken += 1
                continue
            rows[r["key"]] = r  # the same call collected twice is sent once
    if broken:
        print(f"{broken} unreadable lines in {requests_path.name} skipped", flush=True)
    late = harvest(requests_path)
    with judge_api._disk_cache_lock:
        judge_api._load_cache()
        late = {k: t for k, t in late.items() if k in rows and k not in judge_api._disk_cache}
        for key, text in late.items():
            judge_api._disk_cache[key] = text
            judge_api._save_entry(key, text)
        cached = {k for k in rows if k in judge_api._disk_cache}
    if late:
        print(f"{len(late)} verdicts from batches finished after they were given up", flush=True)
    # chunks are cut from every call, cached or not, so a rerun gives each chunk the same calls (and batch-id file)
    by_model: dict[str, list[dict]] = {}
    for r in rows.values():
        by_model.setdefault(r["model"], []).append(r)
    print(f"{len(rows) - len(cached)} uncached calls: " + ", ".join(
        f"{m} {sum(r['key'] not in cached for r in v)}" for m, v in by_model.items()), flush=True)  # fmt: skip
    todo = []
    for model, rs in sorted(by_model.items()):
        for i in range(0, len(rs), CHUNK):
            k = len(todo)
            todo.append((k, model, rs[i : i + CHUNK]))

    def one(k: int, model: str, chunk: list[dict]) -> None:
        open_ = {r["key"]: r["body"] for r in chunk if r["key"] not in cached}
        if not open_:
            return
        size = "" if CHUNK == 5000 else f".c{CHUNK}"
        state = requests_path.with_name(f"{requests_path.stem}{size}.chunk{k}.batches.json")
        texts = complete(model, open_, state)
        with judge_api._disk_cache_lock:
            for key, text in texts.items():
                if key in open_:
                    judge_api._disk_cache[key] = text
                    judge_api._save_entry(key, text)
        print(f"chunk {k} ({model}): {sum(key in open_ for key in texts)}/{len(open_)} cached", flush=True)

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        for f in [pool.submit(one, *t) for t in todo]:
            f.result()


if __name__ == "__main__":
    app()
