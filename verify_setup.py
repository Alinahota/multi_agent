#!/usr/bin/env python3
"""
Pre-flight check — run this BEFORE main.py to confirm your environment is ready.

Usage:
    python verify_setup.py

It checks:
  1. Python version (>= 3.10)
  2. All required packages are installed
  3. OPENAI_API_KEY is set (from .env or shell environment)
  4. Each data-gathering tool returns real data (DuckDuckGo, Wikipedia, Yahoo Finance)
  5. OpenAI API key is valid (makes one tiny test call)

If everything prints OK, you're ready to run:
    python main.py "Nvidia"
"""

import sys
import os

# ── 1. Python version ──────────────────────────────────────────────────────────
print("=" * 55)
print("  Multi-Agent CI System — Environment Verification")
print("=" * 55)

py = sys.version_info
print(f"\n[1] Python version: {sys.version.split()[0]}", end="  ")
if py >= (3, 10):
    print("✓")
else:
    print("✗  Python 3.10 or newer is required.")
    sys.exit(1)

# ── 2. Package imports ─────────────────────────────────────────────────────────
print("\n[2] Checking package imports …")
required = {
    "crewai":      "crewai",
    "crewai_tools":"crewai-tools",
    "ddgs":        "ddgs",
    "wikipedia":   "wikipedia",
    "yfinance":    "yfinance",
    "dotenv":      "python-dotenv",
}

all_ok = True
for module, pkg in required.items():
    try:
        __import__(module)
        print(f"    {pkg:<25} ✓")
    except ImportError:
        print(f"    {pkg:<25} ✗  (run: pip install {pkg})")
        all_ok = False

if not all_ok:
    print("\n  Some packages are missing. Run:  pip install -r requirements.txt")
    sys.exit(1)

# ── 3. Load .env and check API key ─────────────────────────────────────────────
print("\n[3] Checking OPENAI_API_KEY …")
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("    ✗  OPENAI_API_KEY is not set.")
    print("       Steps to fix:")
    print("         1. Copy .env.example to .env")
    print("         2. Replace 'your_openai_api_key_here' with your real key")
    print("         3. Get a key at: https://platform.openai.com/api-keys")
    sys.exit(1)
elif api_key == "your_openai_api_key_here":
    print("    ✗  You have not replaced the placeholder key in .env")
    sys.exit(1)
else:
    masked = api_key[:7] + "..." + api_key[-4:]
    print(f"    ✓  Key found: {masked}")

# ── 4. Tool connectivity tests (no LLM call) ───────────────────────────────────
print("\n[4] Testing data-gathering tools …")

# DuckDuckGo
try:
    from ddgs import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text("Nvidia company", max_results=2))
    if results:
        print(f"    DuckDuckGo search    ✓  ({len(results)} results)")
    else:
        print("    DuckDuckGo search    ⚠  returned 0 results (may be rate-limited)")
except Exception as e:
    print(f"    DuckDuckGo search    ✗  {e}")

# Wikipedia
try:
    import wikipedia
    page = wikipedia.page("Nvidia", auto_suggest=False)
    print(f"    Wikipedia lookup     ✓  (fetched: {page.title!r})")
except Exception as e:
    print(f"    Wikipedia lookup     ✗  {e}")

# Yahoo Finance
try:
    import yfinance as yf
    info = yf.Ticker("NVDA").info
    if info and info.get("longName"):
        print(f"    Yahoo Finance        ✓  ({info['longName']})")
    else:
        print("    Yahoo Finance        ⚠  returned empty info")
except Exception as e:
    print(f"    Yahoo Finance        ✗  {e}")

# ── 5. OpenAI API key validation (direct REST call) ───────────────────────────
print("\n[5] Validating OpenAI API key (REST call to OpenAI API) …")
try:
    import urllib.request, json as _json, urllib.error

    url     = "https://api.openai.com/v1/chat/completions"
    payload = _json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Reply with exactly the word: OK"}],
        "max_tokens": 5,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data  = _json.loads(r.read())
    reply = data["choices"][0]["message"]["content"].strip()
    print(f"    OpenAI API           ✓  (response: {reply!r})")

except urllib.error.HTTPError as e:
    body = e.read().decode(errors="replace")
    if e.code == 401:
        print("    OpenAI API           ✗  Invalid API key (HTTP 401).")
        print("       Get a key at: https://platform.openai.com/api-keys")
        sys.exit(1)
    elif e.code == 429:
        print("    OpenAI API           ⚠  Rate-limited (key is valid, retry shortly).")
    else:
        print(f"    OpenAI API           ✗  HTTP {e.code}: {body[:200]}")
        sys.exit(1)
except Exception as e:
    print(f"    OpenAI API           ✗  {e}")
    sys.exit(1)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  All checks passed.  You are ready to run:")
print()
print('      python main.py "Nvidia"')
print()
print("  Optional flags:")
print("      --sequential   (disable parallel workers, safer on slow machines)")
print("      --help         (show all options)")
print("=" * 55 + "\n")
