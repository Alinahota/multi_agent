"""
Supervisor / Orchestrator — Multi-Agent Competitive Intelligence System.

Topology : Supervisor-Worker (programmatic state-machine).

  • Reflection pattern      Brief is validated by a
                            deterministic oracle BEFORE being returned; if sections
                            are missing the Synthesis Agent is given feedback and
                            asked to revise (generate → validate → revise loop).
  • Adaptive retries       Each retry uses a different
                            search strategy, not just a repeat of the same query.
  • Failure-mode taxonomy  Failures are classified as
                            REFUSAL | SILENT_WRONG | LOOP_NONCONVERGE | DATA_UNAVAILABLE.
  • LLM-as-judge ready     Validation oracle is deterministic
                            (code-based); an LLM judge could replace / augment it.
  • Parallelisation        Research + Financial run concurrently
                            via ThreadPoolExecutor.

States:  INIT → ROUTING → AGGREGATING → SYNTHESIS → VALIDATING → DONE
Workers do NOT communicate with each other — only with the Supervisor.
"""

import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from crewai import Crew, Process

from agents import create_financial_agent, create_research_agent, create_synthesis_agent
from tasks import create_financial_task, create_research_task, create_synthesis_task

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_ATTEMPTS      = 3   # Retries per worker
MIN_RESULT_LEN    = 80  # Characters below this → result is rejected
MAX_REFLECTION    = 2   # How many reflection/revision cycles for Synthesis
PHASE_COOLDOWN    = 65  # Seconds to pause between phases so rate-limit window resets

# The 6 required sections (oracle checks these are present in the final brief)
REQUIRED_SECTIONS = [
    ("## 1", "company overview"),
    ("## 2", "products"),
    ("## 3", "financial"),
    ("## 4", "competitors"),
    ("## 5", "news"),
    ("## 6", "strategic"),
]


# ── Failure-mode taxonomy (evaluation_slides.pptx) ────────────────────────────

class FailureMode:
    REFUSAL          = "REFUSAL"           # LLM refused / said it can't
    SILENT_WRONG     = "SILENT_WRONG"      # Ran but output is empty / too short
    LOOP_NONCONVERGE = "LOOP_NONCONVERGE"  # Hit max iterations without finishing
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"  # External API / source returned nothing


def extract_retry_delay(exc: Exception) -> float:
    """
    Parse 'Please retry in Xs.' from a 429 error message.
    Returns the suggested wait in seconds, or a safe default of 65 s.
    """
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 5   # add 5 s buffer
    return 65.0


def classify_failure(exc: Exception) -> str:
    msg = str(exc).lower()
    if any(k in msg for k in ("refus", "cannot", "unable", "i'm sorry", "i cannot")):
        return FailureMode.REFUSAL
    if any(k in msg for k in ("timeout", "max iter", "loop", "converge")):
        return FailureMode.LOOP_NONCONVERGE
    if any(k in msg for k in ("short", "empty", "insufficient", "no result")):
        return FailureMode.SILENT_WRONG
    return FailureMode.DATA_UNAVAILABLE


# ── Run statistics (printed in the final log banner) ──────────────────────────

@dataclass
class RunStats:
    retries_used:    dict[str, int]   = field(default_factory=dict)
    fallbacks_used:  list[str]        = field(default_factory=list)
    failure_modes:   list[tuple]      = field(default_factory=list)
    reflection_iters: int             = 0

    def summary(self) -> str:
        lines = ["", "── Run Statistics ──────────────────────────────────"]
        for agent, retries in self.retries_used.items():
            lines.append(f"  {agent:30s} retries used: {retries}/{MAX_ATTEMPTS}")
        if self.fallbacks_used:
            lines.append(f"  Fallbacks triggered: {', '.join(self.fallbacks_used)}")
        else:
            lines.append("  Fallbacks triggered: none")
        if self.failure_modes:
            for agent, mode in self.failure_modes:
                lines.append(f"  {agent} failure mode: {mode}")
        lines.append(f"  Reflection iterations (Synthesis): {self.reflection_iters}")
        lines.append("────────────────────────────────────────────────────")
        return "\n".join(lines)


# ── Supervisor ─────────────────────────────────────────────────────────────────

class CISupervisor:
    """
    Programmatic state-machine Supervisor for the Supervisor-Worker topology.

    State machine:
        INIT → ROUTING → AGGREGATING → SYNTHESIS → VALIDATING → DONE

    The Supervisor is the only entity that communicates with workers.
    Workers (Research Agent, Financial Agent, Synthesis Agent) are isolated:
    they receive tasks from the Supervisor and return results to it.
    """

    def __init__(self, company_name: str, sequential: bool = False, human_in_the_loop: bool = False):
        self.company_name      = company_name
        self.sequential        = sequential        # True → run workers one-by-one
        self.human_in_the_loop = human_in_the_loop # True → pause for analyst review before synthesis
        self.state             = "INIT"
        self.results: dict[str, str] = {}
        self.stats             = RunStats()

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self) -> str:
        self._log_banner(f"CI Analysis started for: '{self.company_name}'")

        # ── STATE: ROUTING ────────────────────────────────────────────────────
        self._set_state("ROUTING")
        mode_label = "sequential" if self.sequential else "parallel"
        logger.info(f"[SUPERVISOR] Dispatching Research + Financial workers ({mode_label}) …")

        if self.sequential:
            # Sequential fallback — safer on machines with rate-limited APIs
            for key, worker_fn, fallback_fn in [
                ("research",  self._research_worker,  self._research_fallback),
                ("financial", self._financial_worker, self._financial_fallback),
            ]:
                self._dispatch_one(key, worker_fn, fallback_fn)
        else:
            # Parallel dispatch — faster, matches the SVG architecture diagram
            with ThreadPoolExecutor(max_workers=2) as pool:
                future_map: dict[Future, str] = {
                    pool.submit(
                        self._run_with_retry,
                        worker_fn=self._research_worker,
                        agent_name="Research Agent",
                        fallback_fn=self._research_fallback,
                    ): "research",
                    pool.submit(
                        self._run_with_retry,
                        worker_fn=self._financial_worker,
                        agent_name="Financial Agent",
                        fallback_fn=self._financial_fallback,
                    ): "financial",
                }

                for future in as_completed(future_map):
                    key = future_map[future]
                    self._collect_future(key, future)

        # ── STATE: AGGREGATING ────────────────────────────────────────────────
        self._set_state("AGGREGATING")
        aggregated = self._aggregate()
        logger.info(f"[SUPERVISOR] Aggregated context ready ({len(aggregated)} chars)")

        # ── HUMAN-IN-THE-LOOP checkpoint (enabled with --hitl flag) ──────────
        if self.human_in_the_loop:
            aggregated = self._hitl_review(aggregated)

        # ── COOLDOWN before Synthesis: let the per-minute rate-limit window reset ─
        logger.info(
            f"[SUPERVISOR] Pausing {PHASE_COOLDOWN}s before Synthesis to clear "
            "rate-limit window …"
        )
        time.sleep(PHASE_COOLDOWN)

        # ── STATE: SYNTHESIS + VALIDATING (Reflection loop) ──────────────────
        # Implements the Reflection pattern 
        # generate → validate with oracle → revise if sections are missing
        self._set_state("SYNTHESIS")
        brief = self._synthesis_with_reflection(aggregated)

        # ── STATE: DONE ───────────────────────────────────────────────────────
        self._set_state("DONE")
        self._log_banner("Brief generation complete")
        logger.info(self.stats.summary())
        return brief

    # ── Adaptive retry / fallback ─────────────────────────────────────────────

    def _run_with_retry(
        self,
        worker_fn: Callable[[int], str],   # fn(attempt) → str
        agent_name: str,
        fallback_fn: Optional[Callable[[], str]] = None,
    ) -> str:
        """
        Retry mechanism (Assignment §3.3):
          • Up to MAX_ATTEMPTS tries with exponential back-off.
          • The attempt number is passed to worker_fn so it can use
            ALTERNATIVE SEARCH STRATEGIES on each retry (adaptive retry,
            aligned with the task-decomposition / reflection lecture content).
          • Failures are classified by FailureMode taxonomy.
          • On full exhaustion, activates fallback_fn.

        Example (adaptive retry):
          Attempt 1: search '{company} overview history'
          Attempt 2: search '{company} Wikipedia annual report'   ← different strategy
          Attempt 3: search '{company} business model crunchbase' ← different strategy
          → all fail → fallback: Wikipedia-only lookup
        """
        last_exc: Optional[Exception] = None
        attempts_used = 0

        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts_used = attempt
            try:
                logger.info(f"[RETRY] {agent_name} — attempt {attempt}/{MAX_ATTEMPTS}")
                result = worker_fn(attempt)

                if not result or len(result.strip()) < MIN_RESULT_LEN:
                    raise ValueError(
                        f"Result too short ({len(result.strip())} chars < {MIN_RESULT_LEN}) "
                        f"[{FailureMode.SILENT_WRONG}]"
                    )

                logger.info(f"[RETRY] {agent_name} — succeeded on attempt {attempt} ✓")
                self.stats.retries_used[agent_name] = attempt
                return result

            except Exception as exc:
                last_exc = exc
                mode = classify_failure(exc)
                logger.warning(
                    f"[RETRY] {agent_name} — attempt {attempt} failed [{mode}]: {exc}"
                )
                if attempt < MAX_ATTEMPTS:
                    err_str = str(exc)
                    if "429" in err_str or "quota" in err_str.lower():
                        # Respect the API-recommended retry delay (free-tier = ~45s/min window)
                        wait = extract_retry_delay(exc)
                        logger.info(f"[RETRY] Rate-limited — waiting {wait:.0f}s (API-recommended) …")
                    else:
                        wait = 5 * attempt   # 5s, 10s for non-rate-limit errors
                        logger.info(f"[RETRY] Back-off {wait}s before next attempt …")
                    time.sleep(wait)

        self.stats.retries_used[agent_name] = attempts_used

        # ── FALLBACK ──────────────────────────────────────────────────────────
        logger.error(
            f"[FALLBACK] {agent_name} exhausted {MAX_ATTEMPTS} attempts. "
            "Activating fallback strategy."
        )
        self.stats.fallbacks_used.append(agent_name)
        if fallback_fn:
            return fallback_fn()

        raise RuntimeError(
            f"{agent_name} failed after {MAX_ATTEMPTS} attempts. No fallback provided."
        ) from last_exc

    # ── Worker functions (accept attempt# → adaptive query selection) ──────────

    def _research_worker(self, attempt: int) -> str:
        """
        Adaptive research queries per attempt (lecture: each retry tries a different angle).
          Attempt 1 — broad overview + Wikipedia
          Attempt 2 — news-focused + annual report angle
          Attempt 3 — Crunchbase / business-press angle
        """
        query_variants = {
            1: f"{self.company_name} company overview products history",
            2: f"{self.company_name} annual report 2024 business news latest",
            3: f"{self.company_name} business model products Crunchbase",
        }
        hint = query_variants.get(attempt, query_variants[1])
        logger.info(f"[Research Worker] Attempt {attempt} — query hint: {hint!r}")

        agent = create_research_agent(self.company_name)
        task  = create_research_task(agent, self.company_name, query_hint=hint)
        crew  = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        return str(crew.kickoff())

    def _financial_worker(self, attempt: int) -> str:
        """
        Adaptive financial queries per attempt.
          Attempt 1 — Yahoo Finance primary, DuckDuckGo for ticker
          Attempt 2 — revenue / valuation press angle
          Attempt 3 — SEC filings / investor relations angle
        """
        query_variants = {
            1: f"{self.company_name} stock ticker revenue market cap",
            2: f"{self.company_name} revenue 2024 annual earnings valuation",
            3: f"{self.company_name} investor relations SEC filing financial results",
        }
        hint = query_variants.get(attempt, query_variants[1])
        logger.info(f"[Financial Worker] Attempt {attempt} — query hint: {hint!r}")

        agent = create_financial_agent(self.company_name)
        task  = create_financial_task(agent, self.company_name, query_hint=hint)
        crew  = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        return str(crew.kickoff())

    # ── Reflection / validation loop (Reflection pattern, Module 1 Slides 34-35) ──

    def _synthesis_with_reflection(self, aggregated: str) -> str:
        """
        Implements the Reflection design pattern from the lecture:
          1. Synthesis Agent generates the brief.
          2. A deterministic oracle (code) validates that all 6 sections are present.
          3. If sections are missing, the agent receives specific feedback and revises.
          4. Repeat up to MAX_REFLECTION times, then return best effort.

        This mirrors the 'Critic Agent' loop shown in the lecture (Slides 34-35),
        except the critic here is a rule-based oracle (not a second LLM),
        which the evaluation lecture calls an 'objective eval'.
        """
        context     = aggregated
        brief       = ""
        prev_brief  = ""

        for iteration in range(1, MAX_REFLECTION + 2):   # +2: initial + MAX_REFLECTION revisions
            self._set_state(f"SYNTHESIS-iter{iteration}")
            logger.info(f"[REFLECTION] Synthesis iteration {iteration} …")

            agent  = create_synthesis_agent()
            task   = create_synthesis_task(agent, self.company_name, context)
            crew   = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
            brief  = str(crew.kickoff())

            # ── Oracle validation (deterministic check, evaluation_slides.pptx) ──
            missing = self._oracle_check(brief)

            if not missing:
                logger.info(f"[REFLECTION] Oracle check passed ✓ on iteration {iteration}")
                self.stats.reflection_iters = iteration
                return brief

            logger.warning(
                f"[REFLECTION] Oracle check failed — missing sections: {missing}"
            )
            self.stats.reflection_iters = iteration

            if iteration <= MAX_REFLECTION:
                # Provide feedback to the next iteration (Reflection pattern)
                feedback = (
                    f"\n\nPREVIOUS DRAFT (incomplete — do NOT copy verbatim):\n{brief}\n\n"
                    f"ORACLE FEEDBACK: The following sections are MISSING or INCOMPLETE "
                    f"and MUST be added in your revised output:\n"
                    + "\n".join(f"  • {s}" for s in missing)
                )
                context = aggregated + feedback
                logger.info("[REFLECTION] Feeding oracle feedback back to Synthesis Agent …")
            else:
                logger.warning(
                    f"[REFLECTION] Max reflection iterations ({MAX_REFLECTION}) reached. "
                    "Returning best-effort brief."
                )

        return brief   # best effort

    def _oracle_check(self, brief: str) -> list[str]:
        """
        Deterministic oracle (objective eval, evaluation_slides.pptx):
        Returns list of missing section names; empty list = pass.

        Checks that each of the 6 required sections is present in the brief
        using both the heading number and a keyword from the section title.
        """
        brief_lower = brief.lower()
        missing = []
        for heading_num, keyword in REQUIRED_SECTIONS:
            if heading_num.lower() not in brief_lower or keyword not in brief_lower:
                missing.append(f"{heading_num} ({keyword})")
        return missing

    # ── Human-in-the-loop checkpoint ──────────────────────────────────────────

    def _hitl_review(self, aggregated: str) -> str:
        """
        Human-in-the-loop checkpoint (Assignment §3.3 Bonus):
        Pauses after aggregation, shows the financial/competitor summary to a
        human analyst, and lets them confirm or correct before synthesis begins.

        Enabled with:  python main.py "Nvidia" --hitl
        """
        bar = "=" * 60
        print(f"\n{bar}")
        print("  [HUMAN-IN-THE-LOOP] Competitor Review Checkpoint")
        print(bar)
        print(f"\n  System has gathered intelligence on: {self.company_name}")
        print("  Review the financial and competitor data below before synthesis.\n")

        fin_start = aggregated.find("FINANCIAL INTELLIGENCE")
        preview = aggregated[fin_start:fin_start + 1000] if fin_start >= 0 else aggregated[:1000]
        print(preview)

        print(f"\n{'-' * 60}")
        print("  Press Enter to confirm and proceed to synthesis,")
        print("  or type a correction / addition and press Enter:")
        try:
            correction = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            correction = ""

        if correction:
            logger.info(f"[HITL] Human correction received: {correction!r}")
            aggregated += (
                f"\n\nHUMAN ANALYST CORRECTION / ADDITION:\n{correction}\n"
                "(Incorporate this correction when writing the brief.)"
            )
            print("  [HITL] Correction noted. Incorporating into synthesis context.")
        else:
            logger.info("[HITL] Human confirmed — no corrections.")
            print("  [HITL] Confirmed. Proceeding to synthesis.")

        print(f"{bar}\n")
        return aggregated

    # ── Fallback strategies  ─────────────────────────────────

    def _research_fallback(self) -> str:
        """
        Fallback: Wikipedia-only lookup when the full Research Agent fails.

        Concrete example: DuckDuckGo rate-limits requests during a run on
        an obscure company like 'Cerebras Systems'. The agent cannot complete
        its web-search steps. This fallback fetches the Wikipedia article directly,
        providing foundational background data rather than a total failure.
        """
        logger.info(f"[FALLBACK] Research → Wikipedia-only for '{self.company_name}'")
        from tools import WikipediaTool
        wiki = WikipediaTool()
        try:
            content = wiki._run(self.company_name)
            return (
                "[FALLBACK — Wikipedia only; full web search unavailable]\n\n"
                + content
            )
        except Exception as exc:
            logger.error(f"[FALLBACK] Wikipedia also failed: {exc}")
            return (
                f"[FALLBACK — All research sources unavailable]\n"
                f"Could not retrieve any research data for '{self.company_name}'. "
                f"Error: {exc}. "
                f"Synthesis Agent must acknowledge this gap in the brief."
            )

    def _financial_fallback(self) -> str:
        """
        Fallback: DuckDuckGo revenue/competitor search when Yahoo Finance fails.

        Concrete example: '{self.company_name}' is a private company (e.g., Stripe),
        so it has no ticker on Yahoo Finance. The Financial Agent's yfinance tool
        returns no data after 3 attempts. This fallback searches for press-reported
        revenue estimates and competitor landscape, labelling all figures as ESTIMATED.
        """
        logger.info(
            f"[FALLBACK] Financial → DuckDuckGo estimates for '{self.company_name}'"
        )
        from tools import DuckDuckGoTool
        ddg = DuckDuckGoTool()
        try:
            revenue    = ddg._run(f"{self.company_name} revenue 2024 annual financial results")
            competitors = ddg._run(f"{self.company_name} top competitors 2024 market share")
            return (
                "[FALLBACK — Search-only estimates; all figures should be labeled ESTIMATED]\n\n"
                f"REVENUE / VALUATION ESTIMATES:\n{revenue}\n\n"
                f"COMPETITOR LANDSCAPE:\n{competitors}"
            )
        except Exception as exc:
            logger.error(f"[FALLBACK] DuckDuckGo financial fallback also failed: {exc}")
            return (
                f"[FALLBACK — Financial data completely unavailable]\n"
                f"'{self.company_name}' financial data could not be retrieved. "
                f"The company may be private or APIs are unavailable. "
                f"Consult Bloomberg or SEC EDGAR. Error: {exc}"
            )

    # ── Aggregation ────────────────────────────────────────────────────────────

    def _aggregate(self) -> str:
        return (
            f"RESEARCH INTELLIGENCE — {self.company_name}\n"
            f"{'─' * 60}\n"
            f"{self.results.get('research', 'Research data not available.')}\n\n"
            f"FINANCIAL INTELLIGENCE — {self.company_name}\n"
            f"{'─' * 60}\n"
            f"{self.results.get('financial', 'Financial data not available.')}\n"
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _dispatch_one(
        self,
        key: str,
        worker_fn: Callable[[int], str],
        fallback_fn: Callable[[], str],
    ) -> None:
        """Run a single worker synchronously and store its result."""
        agent_name = key.replace("_", " ").title() + " Agent"
        try:
            self.results[key] = self._run_with_retry(
                worker_fn=worker_fn,
                agent_name=agent_name,
                fallback_fn=fallback_fn,
            )
            logger.info(f"[SUPERVISOR] ✓ {agent_name} done ({len(self.results[key])} chars)")
        except Exception as exc:
            mode = classify_failure(exc)
            logger.error(f"[SUPERVISOR] ✗ {agent_name} CRITICAL FAILURE [{mode}]: {exc}")
            self.stats.failure_modes.append((agent_name, mode))
            self.results[key] = (
                f"[CRITICAL FAILURE — {mode}] "
                f"{agent_name} data unavailable after all retries and fallback. "
                f"Error: {exc}"
            )

    def _collect_future(self, key: str, future: "Future[str]") -> None:
        """Collect result from a completed ThreadPoolExecutor future."""
        agent_name = key.replace("_", " ").title() + " Agent"
        try:
            self.results[key] = future.result()
            logger.info(f"[SUPERVISOR] ✓ {agent_name} done ({len(self.results[key])} chars)")
        except Exception as exc:
            mode = classify_failure(exc)
            logger.error(f"[SUPERVISOR] ✗ {agent_name} CRITICAL FAILURE [{mode}]: {exc}")
            self.stats.failure_modes.append((agent_name, mode))
            self.results[key] = (
                f"[CRITICAL FAILURE — {mode}] "
                f"{agent_name} data unavailable after all retries and fallback. "
                f"Error: {exc}"
            )

    def _set_state(self, new_state: str) -> None:
        logger.info(f"[SUPERVISOR] State: {self.state} → {new_state}")
        self.state = new_state

    @staticmethod
    def _log_banner(msg: str) -> None:
        bar = "=" * 60
        logger.info(f"\n{bar}\n  {msg}\n{bar}")
