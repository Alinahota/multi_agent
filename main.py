#!/usr/bin/env python3
"""
Multi-Agent Competitive Intelligence System
==========================================
Framework : CrewAI 1.x
LLM       : GPT-4o-mini (OpenAI)
Topology  : Supervisor-Worker (programmatic state machine)

Usage:
    python main.py "Nvidia"
    python main.py "Tesla" --sequential
    python main.py --help

Flags:
    --sequential    Run Research and Financial agents one after the other
                    instead of in parallel. Use this if you hit rate-limit
                    errors or 429 responses from OpenAI.
    --hitl          Enable human-in-the-loop checkpoint. The system will pause
                    after gathering data and show you the competitor/financial
                    summary before synthesis. You can confirm or type a correction.
    --help          Show this message and exit.

Before running, verify your environment:
    python verify_setup.py
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# ── Load .env FIRST so os.getenv works everywhere below ───────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Logging: console + timestamped file ───────────────────────────────────────
_ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = f"ci_run_{_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers so retry/fallback events stay readable
for _noisy in ("httpx", "httpcore", "openai", "anthropic", "litellm", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ── Helpers ────────────────────────────────────────────────────────────────────

def show_help() -> None:
    print(__doc__)
    sys.exit(0)


def validate_env() -> None:
    """Fail fast with a clear message if OPENAI_API_KEY is missing."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key == "your_openai_api_key_here":
        print("\n[ERROR] OPENAI_API_KEY is not set (or still has the placeholder value).")
        print()
        print("  To fix this:")
        print("    1. Get a key at https://platform.openai.com/api-keys")
        print("    2. Copy .env.example → .env")
        print("    3. Replace 'your_openai_api_key_here' with your real key")
        print("    4. Re-run:  python main.py \"Nvidia\"")
        print()
        print("  Or run the pre-flight checker first:")
        print("    python verify_setup.py")
        print()
        sys.exit(1)


def parse_args() -> tuple[str, bool, bool]:
    """
    Returns (company_name, use_sequential, human_in_the_loop).
    Accepts:  python main.py "Company Name" [--sequential] [--hitl] [--help]
    """
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        show_help()

    sequential        = "--sequential" in args
    human_in_the_loop = "--hitl" in args
    name_parts        = [a for a in args if not a.startswith("--")]

    if not name_parts:
        print("[ERROR] No company name provided.")
        print('Usage:  python main.py "Nvidia"')
        sys.exit(1)

    return " ".join(name_parts).strip(), sequential, human_in_the_loop


def save_brief(company_name: str, brief: str) -> Path:
    """Write the Competitive Intelligence Brief to a Markdown file."""
    safe  = company_name.replace(" ", "_").replace("/", "-")
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    path  = Path(f"CI_Brief_{safe}_{ts}.md")
    path.write_text(
        f"# Competitive Intelligence Brief: {company_name}\n\n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"**System:** Multi-Agent CI · CrewAI 1.x · GPT-4o-mini (OpenAI) · Supervisor-Worker  \n\n"
        "---\n\n"
        + brief,
        encoding="utf-8",
    )
    return path


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    company_name, sequential, human_in_the_loop = parse_args()
    validate_env()

    print("\n" + "=" * 60)
    print("  Multi-Agent Competitive Intelligence System")
    print(f"  Company    : {company_name}")
    print(f"  Workers    : {'sequential' if sequential else 'parallel'}")
    print(f"  HITL mode  : {'enabled (--hitl)' if human_in_the_loop else 'disabled'}")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log file   : {_log_file}")
    print("=" * 60)
    print()
    print("  Tip: if you see 429 rate-limit errors, re-run with --sequential")
    print()

    # Import here — after env is loaded, so LLM config picks up OPENAI_API_KEY
    from supervisor import CISupervisor

    supervisor = CISupervisor(company_name, sequential=sequential, human_in_the_loop=human_in_the_loop)
    brief = supervisor.run()

    # ── Save output ────────────────────────────────────────────────────────────
    out_path = save_brief(company_name, brief)

    print("\n" + "=" * 60)
    print(f"  Brief saved to : {out_path}")
    print(f"  Log saved to   : {_log_file}")
    print("=" * 60 + "\n")
    print(brief)


if __name__ == "__main__":
    main()
