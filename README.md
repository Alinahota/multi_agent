# Multi-Agent Competitive Intelligence System

**Course:** Leveraging LLM for Productivity Improvement— MSBA Program, McCombs School of Business  
**Framework:** CrewAI 1.9.3 | **LLM:** GPT-4o-mini (OpenAI) | **Topology:** Supervisor-Worker

---

## Quick Start (Professor Grading)

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Open .env and replace  your_openai_api_key_here  with your actual key

# 4. Verify everything is set up correctly
python verify_setup.py

# 5. Run the system
python main.py "Nvidia"
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | **3.10 or newer** |
| pip | any recent version |
| OpenAI API key | paid account (~$0.05 per full run) |

### Getting an OpenAI API Key

1. Go to **https://platform.openai.com/api-keys**
2. Sign in and click **Create new secret key**
3. Copy the key 
4. Paste it into your `.env` file (see step 3 above)
5. Ensure your account has at least $1 credit (one full run costs ~$0.05 with GPT-4o-mini)

---

## Installation

```bash
pip install -r requirements.txt
```

This installs:
- `crewai==1.9.3` — multi-agent orchestration framework
- `crewai-tools==1.9.3` — tool base classes
- `ddgs` — DuckDuckGo web search (free, no API key)
- `wikipedia` — Wikipedia article fetcher (free)
- `yfinance` — Yahoo Finance financial data (free)
- `python-dotenv` — loads `OPENAI_API_KEY` from `.env`

---

## How to Set the API Key

Create a file named `.env` in the `multi_agent_ci/` directory:

```
OPENAI_API_KEY=sk-proj-...your_actual_key_here...
```

A template is provided as `.env.example`. 

---

## How to Run

```bash
# Basic usage
python main.py "Nvidia"

# Other companies
python main.py "Tesla"
python main.py "Salesforce"
python main.py "Chipotle"

# Optional: run workers one-at-a-time instead of in parallel
python main.py "Nvidia" --sequential
```

**Expected runtime:** 2–4 minutes

---

## Expected Output

The system prints the brief to the console and saves two files:

| File | Description |
|---|---|
| `CI_Brief_<Company>_<timestamp>.md` | The full Competitive Intelligence Brief (Markdown) |
| `ci_run_<timestamp>.log` | Execution log including all retry and fallback events |

### Brief structure (six sections, 400–800 words)

```
## 1. Company Overview
## 2. Products & Services
## 3. Financial Snapshot
## 4. Top Competitors
## 5. Recent News & Developments
## 6. Strategic Assessment
```

---

## System Architecture

```
Company Name Input
       │
       ▼
┌──────────────────────────────────────────┐
│   SUPERVISOR (Programmatic State Machine) │
│   States: INIT→ROUTING→AGGREGATING→      │
│           SYNTHESIS→VALIDATING→DONE      │
└──────┬──────────────────────┬────────────┘
       │  (parallel by default)│
       ▼                       ▼
┌──────────────┐       ┌──────────────────┐
│ Research     │       │ Financial        │
│ Agent        │       │ Analyst Agent    │
│              │       │                  │
│ Tools:       │       │ Tools:           │
│ • DuckDuckGo │       │ • Yahoo Finance  │
│ • Wikipedia  │       │ • DuckDuckGo     │
└──────┬───────┘       └────────┬─────────┘
       │  retry ×3 + fallback   │
       └───────────┬────────────┘
                   ▼
       ┌───────────────────────┐
       │ Supervisor aggregates │
       └──────────┬────────────┘
                  ▼
       ┌───────────────────────┐
       │   Synthesis Agent     │  ← no external tools; writes from context
       │   + Oracle Validation │  ← checks all 6 sections present
       │   + Reflection loop   │  ← revises if sections missing
       └──────────┬────────────┘
                  ▼
       Competitive Intelligence Brief
```

**Failure handling:**
- **Retry:** each worker gets up to 3 attempts with exponential back-off. Each retry uses a different search strategy (adaptive retry).
- **Fallback:** if all retries fail:
  - Research fallback → Wikipedia-only lookup
  - Financial fallback → DuckDuckGo revenue/competitor search with ESTIMATED labels

**Reflection (from lecture design patterns):**
- After the Synthesis Agent writes the brief, a deterministic oracle checks that all 6 section headings are present. If any are missing, the agent receives specific feedback and is asked to revise (up to 2 reflection iterations).

---

## Project Files

```
multi_agent_ci/
├── main.py           CLI entry point (argparse, logging setup, output saving)
├── supervisor.py     Supervisor state machine — retry, fallback, reflection, aggregation
├── agents.py         CrewAI agent definitions (roles, goals, backstories, LLM config)
├── tasks.py          Task descriptions with adaptive query hints per retry attempt
├── tools.py          Custom tools: DuckDuckGoTool, WikipediaTool, YFinanceTool
├── verify_setup.py   Pre-flight checker — run before main.py
├── requirements.txt  Pinned dependency list
├── .env.example      API key template (copy to .env and fill in)
└── README.md         This file
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `OPENAI_API_KEY is not set` | Create `.env` from `.env.example` and add your OpenAI key |
| `No data found for ticker` | Company may be private; financial agent uses DuckDuckGo fallback |
| `DuckDuckGo returned no results` | Temporary rate-limit; retry logic retries automatically |
| Import errors | Run `pip install -r requirements.txt` again |
| Python 3.9 or older | Upgrade to Python 3.10+ (`python --version` to check) |

---

## Design Choices (Brief Summary — see write-up for full discussion)

- **Framework:** CrewAI — native support for role-based agents, tool integration, and LLM-agnostic backends via LiteLLM.
- **LLM:** GPT-4o-mini (OpenAI) — cost-efficient (~$0.05/run), first-class CrewAI support, reliable tool-use.
- **Topology:** Supervisor-Worker — natural fit for independent parallel data gathering followed by a single synthesis step.
- **Supervisor type:** Programmatic state machine (not LLM-routed) — deterministic, easier to test and explain.
- **Retry strategy:** Adaptive — each attempt uses a different search query angle (not just a repeat).
- **Fallback strategy:** Different tool/source — Research falls back to Wikipedia; Financial falls back to DuckDuckGo estimates.
- **Reflection:** Oracle-validated synthesis loop ensures all 6 sections are present before returning.
