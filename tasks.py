"""
Task definitions for each worker agent in the CI pipeline.

Each task has a precise description (what to do) and expected_output (what to return).
The Supervisor creates and runs these tasks individually so it can apply retry/fallback
around each one independently.
"""

from crewai import Agent, Task


def create_research_task(
    agent: Agent,
    company_name: str,
    query_hint: str = "",
) -> Task:
    """
    Research Agent task: company overview + products + recent news.

    query_hint: injected by the Supervisor's adaptive retry logic to guide
                the agent toward a different search angle on each attempt.
    """
    hint_block = (
        f"\nADAPTIVE RETRY HINT (use these keywords in at least one search): {query_hint}\n"
        if query_hint else ""
    )
    return Task(
        description=f"""
You are researching {company_name} for a Competitive Intelligence Brief.
{hint_block}
Gather information across THREE areas using your tools:

AREA 1 — COMPANY OVERVIEW
  • Search: "{company_name} company overview history founded headquarters"
  • Search: "{company_name} CEO leadership business model"
  • Also look up "{company_name}" on Wikipedia for authoritative background.
  • Collect: full legal name, founded year, HQ location, CEO, core business model,
    mission/vision statement (if public), approximate headcount.

AREA 2 — PRODUCTS & SERVICES
  • Search: "{company_name} products services portfolio 2024 2025"
  • Identify: 3–6 main product lines or service categories,
    key revenue-generating offerings, any major product launches in the past year.

AREA 3 — RECENT NEWS & DEVELOPMENTS
  • Search: "{company_name} news 2024 2025"
  • Search: "{company_name} earnings partnership acquisition announcement"
  • Collect: 3–5 significant developments from the past 6 months
    (earnings beats/misses, partnerships, M&A, regulatory events, product launches).
  • For EVERY news item, record the publication name and date (e.g., "Bloomberg, January 2025").
    If you cannot identify the publication, write the domain name (e.g., "techcrunch.com, 2025").
    Do NOT include a news item without a source attribution.

RULES:
- Use at least 3 separate search queries across the three areas.
- Do NOT fabricate information. If you cannot find a fact, write "Not found."
- Cite the source (URL or "Wikipedia") for each key claim.
- Report raw findings — the Synthesis Agent will write the polished prose.
""",
        expected_output=f"""
A structured research report for {company_name} with these labeled sections:

[COMPANY OVERVIEW]
<founding year, HQ, CEO, business model, headcount, mission>

[PRODUCTS & SERVICES]
<list of 3–6 main offerings with one-sentence descriptions>

[RECENT NEWS]
<3–5 bullet points, each with: date, headline, 1–2 sentence summary, and source (Publication Name, Date)>

Sources used: <list of URLs or "Wikipedia">
""",
        agent=agent,
    )


def create_financial_task(
    agent: Agent,
    company_name: str,
    query_hint: str = "",
) -> Task:
    """
    Financial Analyst Agent task: financial snapshot + top competitors.

    query_hint: injected by the Supervisor's adaptive retry logic.
    """
    hint_block = (
        f"\nADAPTIVE RETRY HINT (prioritise these keywords in searches): {query_hint}\n"
        if query_hint else ""
    )
    return Task(
        description=f"""
You are producing the financial analysis section of a Competitive Intelligence Brief for {company_name}.
{hint_block}
STEP 1 — FIND THE TICKER (if public)
  • Search: "{company_name} stock ticker symbol NYSE NASDAQ"
  • If you find a ticker, note it for Step 2.
  • If the search suggests the company is private, skip to the fallback in Step 2.

STEP 2 — GATHER FINANCIAL DATA
  Option A (public company): Use the Yahoo Finance tool with the ticker symbol you found.
    Retrieve: market cap, revenue (TTM), net income, gross margin, operating margin,
    revenue growth YoY, P/E ratio, EPS, 52-week high/low, employee count.
  Option B (private company OR ticker lookup failed):
    Search: "{company_name} revenue 2024 annual report"
    Search: "{company_name} valuation funding round"
    Report whatever estimates you find and label them clearly as ESTIMATES.
    Write "Data not available — company is private" for any figure you cannot find.

STEP 3 — IDENTIFY TOP COMPETITORS
  • Search: "{company_name} top competitors 2024 market share"
  • Search: "{company_name} vs competitors comparison"
  • Identify 3–5 direct competitors. For each, note:
      - Company name
      - Why it competes with {company_name} (one sentence)
      - Any rough market share or positioning note if available.

CRITICAL RULES:
- NEVER fabricate financial figures. Use only data returned by your tools.
- Label every figure as: REPORTED (from Yahoo Finance), ESTIMATED (from search), or UNAVAILABLE.
- If financial data is completely unavailable, write: "Financial data unavailable — {company_name} appears to be private."
""",
        expected_output=f"""
A structured financial analysis for {company_name} with these labeled sections:

[FINANCIAL SNAPSHOT]
<Key metrics table: Revenue, Market Cap, Net Income, Gross Margin, Operating Margin,
Revenue Growth, P/E Ratio, EPS, Employees — each labeled REPORTED / ESTIMATED / UNAVAILABLE>
<Note if the company is private>

[TOP COMPETITORS]
<3–5 competitors, each with: name, brief positioning note, any market share data>

Data provenance: <REPORTED from Yahoo Finance / ESTIMATED from web search / UNAVAILABLE>
""",
        agent=agent,
    )


def create_synthesis_task(agent: Agent, company_name: str, aggregated_context: str) -> Task:
    """
    Synthesis Agent task: write the final 6-section Competitive Intelligence Brief.
    The full research + financial context is embedded directly in the task description.
    """
    return Task(
        description=f"""
You are writing a Competitive Intelligence Brief for {company_name}.
You have been given the following raw intelligence gathered by specialist agents.
Use ONLY this information — do not access any external sources or invent data.

══════════════════════════════════════════════════════
INTELLIGENCE PACKAGE FOR {company_name.upper()}
══════════════════════════════════════════════════════
{aggregated_context}
══════════════════════════════════════════════════════

Write a professional Competitive Intelligence Brief with EXACTLY these six sections,
each preceded by its markdown heading:

## 1. Company Overview
Cover: company history, founding, HQ, CEO, business model, size (employees/revenue scale).
Length: 3–5 sentences.

## 2. Products & Services
Cover: main product lines and services, key offerings, recent launches or notable features.
Length: 3–5 sentences.

## 3. Financial Snapshot
Cover: revenue, market cap, growth rate, key margins, P/E ratio if available.
IMPORTANT: Clearly label each figure as REPORTED, ESTIMATED, or UNAVAILABLE.
If the company is private and figures are unavailable, say so — do not fabricate numbers.
Length: 3–5 sentences.

## 4. Top Competitors
Cover: 3–5 direct competitors with brief descriptions of why they compete and positioning.
Length: one short paragraph or 3–5 bullet points.

## 5. Recent News & Developments
Cover: 3–5 significant developments from the past 6 months (earnings, partnerships, launches, M&A).
For EACH item, end with a source attribution in parentheses: (Source: Publication Name, Month Year).
Use the source information provided in the intelligence package. If no source was recorded, write (Source: web search).
Length: 3–5 bullet points, each 1–2 sentences.

## 6. Strategic Assessment
Cover: 2–3 key strengths, 1–2 risks or vulnerabilities, overall strategic positioning.
Be analytical and evidence-based — cite specific facts from the intelligence package.
Length: 3–5 sentences.

FORMATTING RULES:
- Total length: 400–800 words (brief is better than bloated).
- Professional tone suitable for C-suite executives.
- Do NOT start with "Here is the brief" or similar meta-phrases — start directly with ## 1.
- If a section has insufficient data, acknowledge it honestly in one sentence.
""",
        expected_output=f"""
A complete, professional Competitive Intelligence Brief for {company_name} containing
all six labeled sections (## 1 through ## 6), 400–800 words total, with honest
acknowledgment of any data gaps and no fabricated figures.
""",
        agent=agent,
    )
