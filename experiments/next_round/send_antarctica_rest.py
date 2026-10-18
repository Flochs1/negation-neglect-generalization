"""Antarctica generalisation judge calls outside the stalled batches, sent at full price into the judge cache
(project lead 2026-10-15: skip the 25k stuck calls, do the rest until the money runs out).

The stage's calls were cut into 5000-call chunks in file order (judge_batch.run): chunks 0-1 are answered and cached,
chunks 2-6 (calls 10,000-34,999) sit in the stalled batches, and calls 35,000 on were never sent. Those are sent
here, 16 at a time; it stops at the first out-of-credit error.

    uv run python experiments/next_round/send_antarctica_rest.py
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv

load_dotenv(".env")
sys.path.insert(0, ".")
from src.evals import judge_api  # noqa: E402

REQ = "experiments/eval_v2/antarctica/judge_batches/generalisation.jsonl"
URL = "https://openrouter.ai/api/v1/chat/completions"
STOP = threading.Event()

rows = {}
with open(REQ) as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            rows[r["key"]] = r
rs = list(rows.values())
judge_api._load_cache()
cache = judge_api._disk_cache
assert len(rs) == 44269, len(rs)
assert all(r["key"] in cache for r in rs[:10000]), "chunks 0-1 should be cached"
rest = [r for r in rs[35000:] if r["key"] not in cache]
print(f"{len(rest)} calls to send (of {len(rs) - 35000} outside the stalled batches)", flush=True)
headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
done, failed = [0], [0]


def send(r: dict) -> None:
    if STOP.is_set():
        return
    for _ in range(3):
        try:
            resp = httpx.post(URL, headers=headers, json={"model": r["model"], **r["body"]}, timeout=300)
            if resp.status_code == 402:
                STOP.set()
                print(f"out of credit: {resp.text[:200]}", flush=True)
                return
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"].get("content") or ""
            if text.strip():  # stored as judge_api.judge_one stores it: raw, empty answers never
                with judge_api._disk_cache_lock:
                    judge_api._disk_cache[r["key"]] = text
                    judge_api._save_entry(r["key"], text)
                done[0] += 1
                if done[0] % 500 == 0:
                    print(f"{done[0]} answered", flush=True)
                return
        except (httpx.HTTPError, KeyError, ValueError):
            continue
    failed[0] += 1


with ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(send, rest))
print(f"answered {done[0]}, failed {failed[0]}, stopped for credit: {STOP.is_set()}", flush=True)
sys.exit(1 if STOP.is_set() else 0)
