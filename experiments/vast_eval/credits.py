"""Print OpenRouter total usage (USD) -- checked before and after each judged stage."""

import os

import httpx
from dotenv import load_dotenv

load_dotenv(".env")
r = httpx.get(
    "https://openrouter.ai/api/v1/credits",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    timeout=30,
)
print(f"openrouter total_usage {r.json()['data']['total_usage']:.4f}")
