# Multi-Agent Competitive Intelligence System
### Write-Up | Autonomous AI Agents | MSBA Program | McCombs School of Business

---

## Project Overview

This project is a multi-agent AI system that takes a company name as input and automatically produces a structured Competitive Intelligence Brief. The brief covers six areas: company overview, products and services, financial snapshot, top competitors, recent news, and a strategic assessment. Everything from data gathering to final writing is handled by specialized AI agents working together, with no manual intervention needed once the system is started.

The system was built using CrewAI as the orchestration framework, GPT-4o-mini as the language model, and a Supervisor-Worker topology where a central supervisor coordinates three specialized worker agents. Data is gathered from free, real-world sources including DuckDuckGo web search, Wikipedia, and Yahoo Finance. The system was tested on two companies: Nvidia Corporation (a large public company with full Yahoo Finance data) and Stripe (a private fintech company with no public listing, used to test failure handling). Both full output briefs are included as Deliverable 4.

---

## 1. Design Decisions

**Framework Choice: CrewAI**

CrewAI was chosen because it provides a clean way to define agents with distinct roles, goals, and tools, and it has first-class support for OpenAI models. It handles the low-level details of LLM calls and tool execution, so the focus could stay on the architecture rather than plumbing. Compared to raw LangChain, CrewAI requires less boilerplate for multi-agent setups. Compared to AutoGen, it gives more explicit control over how agents are defined and what they can access.

**LLM Choice: GPT-4o-mini**

GPT-4o-mini was chosen for its reliability with tool use, reasonable cost (roughly $0.05 per full run), and strong performance on structured writing tasks. The free-tier Gemini API was initially considered but consistently hit per-minute rate limits during testing, which would have made the system unreliable for grading. GPT-4o-mini handled all three agents across a full run without rate-limit issues.

**Topology Choice: Supervisor-Worker**

The Supervisor-Worker topology was chosen because the problem naturally splits into independent data-gathering tasks followed by a single synthesis step. The Research Agent and Financial Agent do not need each other's results to do their work, so they can run in parallel. Once both finish, the Supervisor collects their outputs and hands everything to the Synthesis Agent. This is a better fit than Sequential Handoff, which would force the financial agent to wait for the research agent unnecessarily. It is also simpler than Hierarchical, which adds complexity that is not needed with only three workers.

The Supervisor is implemented as a programmatic state machine rather than an LLM-routed supervisor. This means the routing logic is written in Python code, not decided by a language model. The states are: INIT, ROUTING, AGGREGATING, SYNTHESIS, VALIDATING, and DONE. This approach was chosen because it is predictable, easy to debug, and produces clear logs showing exactly what the system did at each step.

**Agent Decomposition**

The problem was split across three agents, each with a focused role:

The **Research Agent** is responsible for gathering factual background on the company. Its goal is to search the web and Wikipedia to find the company's history, headquarters, CEO, business model, main products, and significant news from the past six months. It has access to DuckDuckGo and Wikipedia as tools. In prompt language: *"You never invent facts — if you cannot find something, you say so explicitly."*

The **Financial Analyst Agent** is responsible for the numbers side of the brief. Its goal is to find the stock ticker, pull real financial data from Yahoo Finance, and identify the top three to five competitors. It has access to Yahoo Finance and DuckDuckGo. In prompt language: *"You are rigorous about data provenance: you clearly label figures as 'reported', 'estimated', or 'unavailable'. You never guess or interpolate financial data."*

The **Synthesis Agent** takes everything the other two agents found and writes the final brief. It has no tools and makes no external calls. Its goal is to produce a clean, professional six-section document using only the information passed to it by the Supervisor. In prompt language: *"You never embellish or fill gaps with assumptions — when data is missing you say so and explain why. Your briefs run 400–800 words and follow a strict six-section structure."*

---

## 2. Failure Handling

The system implements two failure-handling mechanisms as required: retry logic and fallback logic. The system was also run on a second company (Stripe, a private fintech company) specifically to test how failure handling behaves when standard data sources return no results.

**Retry Logic**

Each worker agent gets up to three attempts before the system gives up and activates the fallback. What makes these retries meaningful is that each attempt uses a different search strategy rather than just repeating the same query. On the first attempt, the Research Agent searches for general company overview and history. If that fails, the second attempt shifts to annual reports and recent news. If the second attempt also fails, the third attempt uses Crunchbase and business press sources. The same adaptive approach applies to the Financial Agent. Between failed attempts, the system waits for the amount of time the API recommends in the error message, plus a five-second buffer. The retry logic is implemented in `supervisor.py` (lines 196–266) and produces log output in the format:

```
[RETRY] Financial Agent — attempt 1/3
[RETRY] Financial Agent — attempt 1 failed [DATA_UNAVAILABLE]: ...
[RETRY] Back-off 5s before next attempt …
[RETRY] Financial Agent — attempt 2/3
```

**Fallback Logic**

If all three attempts fail, the system activates a fallback instead of crashing. Each agent has its own fallback strategy designed to still return something useful.

For the Research Agent, the fallback skips DuckDuckGo entirely and goes directly to Wikipedia. This is enough to get basic company background even if web search is unavailable. The result is clearly labeled "[FALLBACK - Wikipedia only]" so the Synthesis Agent knows the source was limited.

For the Financial Agent, the fallback skips Yahoo Finance and uses DuckDuckGo to search for press-reported revenue estimates and competitor rankings. Every figure found this way is labeled ESTIMATED so the final brief does not present unverified numbers as facts. The fallback log output looks like:

```
[FALLBACK] Financial Agent exhausted 3 attempts. Activating fallback strategy.
[FALLBACK] Financial → DuckDuckGo estimates for 'Stripe'
```

**Live Test: Stripe (Private Company)**

The system was run on Stripe to test how it handles a company with no public stock listing. Stripe has no Yahoo Finance ticker. The Financial Agent's behavior is visible in the actual run log (`ci_run_20260427_225929.log`):

```
23:00:21  [RETRY] Financial Agent — attempt 1/3
23:00:21  [Financial Worker] Attempt 1 — query hint: 'Stripe stock ticker revenue market cap'
23:00:23  [DuckDuckGo] Searching: 'Stripe stock ticker symbol NYSE NASDAQ'
23:00:28  [DuckDuckGo] Searching: 'Stripe revenue 2024 annual report'
23:00:32  [DuckDuckGo] Searching: 'Stripe valuation funding round'
23:00:43  [RETRY] Financial Agent — succeeded on attempt 1 ✓
```

What this log shows is that when Yahoo Finance produced no data (no ticker exists for Stripe), the Financial Agent adapted at the tool level: it searched for the ticker first, found none, then pivoted to DuckDuckGo revenue and valuation searches. It returned 1,189 characters of content including ESTIMATED labels, which passed the 80-character minimum threshold and satisfied the supervisor without needing supervisor-level retries or fallback. The final brief correctly noted: "Stripe appears to be a private company and does not publicly disclose detailed financial information," and labeled all figures as estimates. This demonstrates graceful degradation: the system produced a useful, clearly-labeled output rather than fabricating data or crashing.

The supervisor-level retry and fallback are designed to activate when a crew execution either throws an exception (API timeout, 429 rate-limit, connection error) or returns fewer than 80 characters. They would be observable in production scenarios such as running during a DuckDuckGo rate-limit window or when an API key is exhausted. Both mechanisms are fully implemented in `supervisor.py`.

In addition to retry and fallback, the system also implements a reflection loop for the Synthesis Agent. After the brief is written, a code-based oracle checks that all six required section headings are present. If any section is missing, the agent is given specific feedback about what is missing and asked to revise. This can repeat up to two times. The Stripe run confirmed this works: `[REFLECTION] Oracle check passed on iteration 1`. This pattern was covered in the lecture as the Reflection design pattern, where an output is checked and fed back to the agent for correction.

**Human-in-the-Loop Checkpoint (Bonus)**

The system also implements an optional human-in-the-loop checkpoint enabled with the `--hitl` flag. After the Research and Financial agents complete their work and before synthesis begins, the Supervisor pauses and displays the aggregated competitor and financial data to the analyst. The analyst can press Enter to confirm and proceed, or type a correction that is appended to the synthesis context. For example, running `python main.py "Nvidia" --hitl` produces:

```
[HUMAN-IN-THE-LOOP] Competitor Review Checkpoint
The Financial Agent has identified the following for Nvidia:
  ... [financial and competitor summary displayed] ...

Review the competitor and financial data above.
Press Enter to proceed, or type a correction to add context:
> AMD market share should be noted as 12%, not 5% per latest IDC report
[HITL] Correction noted. Incorporating into synthesis context...
```

This is implemented as a simple `input()` call inside the Supervisor's AGGREGATING state. While not sophisticated, it gives a human analyst a checkpoint to catch obvious errors in competitor identification before the brief is finalized.

---

## 3. Results

The system was run on two companies: Nvidia Corporation (a large public company) and Stripe (a large private fintech company). The Stripe run was specifically chosen to test how the system behaves when standard financial data sources return no results. The full briefs are included as Deliverable 4.

**Nvidia (Primary Test)**

The **Company Overview** correctly identified Nvidia as a Santa Clara-based GPU company founded in 1993, led by Jensen Huang, with approximately 42,000 employees and a revenue of $215.94 billion. This was accurate.

The **Products and Services** section covered the GeForce GPU line, the RTX Series with ray-tracing capabilities, the CUDA platform, and the newly launched RTX 5090. This was accurate and reasonably detailed.

The **Financial Snapshot** used real data pulled directly from Yahoo Finance and labeled all figures as REPORTED. Revenue of $215.94 billion, market cap of $5.26 trillion, 73.2% year-over-year revenue growth, 71.1% gross margin, and a P/E ratio of 44.30 were all correct as of the time of the run. No numbers were fabricated.

The **Top Competitors** section listed AMD, Intel, Google (TPUs), and Cerebras Systems. This was mostly accurate. However, Intel's listed GPU market share of 68% appears to be incorrect or outdated — the agent pulled this from a web search result that may have been referring to a different metric (CPU market share, not GPU). The brief does not flag this inline, which is a gap: the Synthesis Agent accepted the figure rather than questioning it.

The **Recent News** section only included three items: the RTX 5090 launch in December 2024, the $3.66 trillion market cap milestone in January 2025, and a Forbes workplace ranking. This met the minimum requirement but was thinner than ideal. The agent did not surface more recent events from early 2025, likely because its search queries were broad rather than date-filtered.

The **Strategic Assessment** was reasonable but somewhat surface-level. It identified Nvidia's GPU dominance and revenue growth as strengths, and competition from AMD and Intel as risks. It did not mention export restrictions, geopolitical risks related to chip manufacturing, or the growing threat from in-house AI chips being developed by Amazon and Microsoft, which are significant strategic factors.

**Stripe (Private Company Test)**

Stripe was chosen because it has no public stock listing, which tests the system's ability to handle missing financial data. Yahoo Finance returned no ticker data, so the Financial Agent automatically pivoted to DuckDuckGo for press-reported estimates. The brief correctly noted that "Stripe appears to be a private company and does not publicly disclose detailed financial information," and labeled the $16 billion revenue estimate and $175.6 billion valuation as estimates from external press sources. No data was fabricated.

Compared to Nvidia, the Stripe brief was more limited in its financial section, which is expected and appropriate. The research sections (products, competitors, news) performed similarly well. The system correctly identified PayPal, Square, Adyen, Braintree, and Authorize.Net as competitors and surfaced the $1.1 billion Bridge Network acquisition from early 2025 as the main recent development.

**Overall Assessment**

Both briefs were useful and factually grounded for sections that relied on structured data. The system handled the private company case without fabrication or crashing. The common weakness across both runs was the web search sections: news and strategic assessment depended on broad queries without date filtering, which produced thin or somewhat surface-level content. This is a known limitation of using general-purpose web search tools.

---

## 4. Evaluation Reflection

If this system were being deployed for real business use, evaluating it would go beyond just checking whether it ran successfully. The question is whether the output is actually trustworthy and useful.

**What to measure**

The most important metric is factual accuracy. For any claim in the brief that can be verified, such as revenue figures, founding dates, or employee counts, those should be spot-checked against authoritative sources like SEC filings or the company's investor relations page. A system that produces a confident-sounding but wrong number is worse than one that says "data unavailable." The Intel 68% GPU market share figure in the Nvidia brief is a concrete example of this failure mode: it was accepted and presented without a confidence tag.

The second metric is section completeness. The code already checks this with the oracle, but the oracle only verifies that section headings exist. A more rigorous version would also check that each section has a minimum number of substantive sentences and does not simply say "information not found" in every line.

A third metric is data freshness. News that is six or more months old is not competitive intelligence. A useful evaluation would check whether the news items in section five are within a specific time window, and flag the brief if too many items are outdated.

**Failure modes to test for**

Testing should include at least four scenarios that are known to be hard for this system:

Private companies with no Yahoo Finance data, such as Stripe or SpaceX. These test whether the financial fallback produces something useful and clearly labeled rather than fabricated. The Stripe run showed the system handles this correctly.

Companies with very common names that create disambiguation problems, such as "Apple" (the company versus the fruit) or "Delta" (airline versus other uses). These test whether the research agent correctly identifies the intended entity. The Wikipedia disambiguation warning in the Stripe run ("trying option: 'Stripe (pattern)'") shows this is a real issue the system already encounters.

Obscure companies with limited online presence. These test how the system behaves when all three research retry attempts come back nearly empty, which would actually trigger the supervisor-level fallback.

Companies that have recently gone through major changes, such as a merger or rebranding. These test whether the system picks up current information or surfaces outdated descriptions.

**What success looks like in practice**

A realistic target for a system like this in a business setting would be that a trained analyst can review the brief in five minutes and confirm it is accurate enough to use as a starting point without needing to fact-check every line. That is a different standard than "all information is perfectly correct," which no automated system can guarantee. The goal is to reduce the time an analyst spends on initial research from two hours to fifteen minutes, while clearly flagging anything that needs verification.

The reflection and fallback mechanisms already move the system in this direction by being transparent about data quality. The next improvement would be adding source citations to every claim so a reviewer knows exactly where each piece of information came from.
