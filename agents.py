"""
Agent definitions for the Multi-Agent Competitive Intelligence System.

Three specialized worker agents:
  1. Research Agent      — company overview, products, recent news
  2. Financial Agent     — financials, competitors
  3. Synthesis Agent     — writes the final 6-section CI Brief

All agents share the same LLM but have distinct roles, goals, and tools.
LLM: GPT-4o-mini via OpenAI API (reliable tool-use, no per-minute rate limits).
"""

import os
import logging
from crewai import Agent, LLM
from tools import DuckDuckGoTool, WikipediaTool, YFinanceTool

logger = logging.getLogger(__name__)


def get_llm() -> LLM:
    """Return a configured GPT-4o-mini instance via CrewAI's LiteLLM backend."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file:  OPENAI_API_KEY=sk-..."
        )
    return LLM(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.3,   # Lower temp → more factual, less hallucination
        max_tokens=4096,
    )


# ── Worker Agent factories ─────────────────────────────────────────────────────

def create_research_agent(company_name: str) -> Agent:
    """
    Research Agent: gathers company overview, products/services, and recent news.
    Tools: DuckDuckGo (current web search) + Wikipedia (background).
    """
    return Agent(
        role="Senior Business Research Analyst",
        goal=(
            f"Gather accurate, factual information about {company_name} covering: "
            "company history and overview, main products and services, and significant "
            "news from the past 6 months. Never fabricate data."
        ),
        backstory=(
            "You are an expert business researcher with 15 years of experience in "
            "competitive intelligence. You are known for finding reliable information "
            "quickly using web search and encyclopedic sources. You always cross-reference "
            "multiple sources and clearly flag when information is uncertain or unavailable. "
            "You never invent facts — if you cannot find something, you say so explicitly."
        ),
        tools=[DuckDuckGoTool(), WikipediaTool()],
        llm=get_llm(),
        verbose=True,
        max_iter=5,
        max_retry_limit=2,
    )


def create_financial_agent(company_name: str) -> Agent:
    """
    Financial Analyst Agent: retrieves financial metrics and maps the competitive landscape.
    Tools: YFinance (financial data) + DuckDuckGo (ticker lookup + private-company estimates).
    """
    return Agent(
        role="Senior Financial Analyst",
        goal=(
            f"Produce an accurate financial snapshot for {company_name} and identify "
            f"its top 3–5 competitors with brief positioning notes. "
            "Use real data from tools only — never fabricate numbers. "
            f"If {company_name} is private or data is unavailable, state that explicitly."
        ),
        backstory=(
            "You are a senior equity research analyst who has covered technology and "
            "growth companies for a top-tier investment bank. You specialise in extracting "
            "meaningful financial metrics from public filings and market data. You are "
            "rigorous about data provenance: you clearly label figures as 'reported', "
            "'estimated', or 'unavailable'. You never guess or interpolate financial data."
        ),
        tools=[YFinanceTool(), DuckDuckGoTool()],
        llm=get_llm(),
        verbose=True,
        max_iter=5,
        max_retry_limit=2,
    )


def create_synthesis_agent() -> Agent:
    """
    Synthesis Agent: synthesizes research + financial intelligence into the final CI Brief.
    No tools — works entirely from context provided by the Supervisor.
    """
    return Agent(
        role="Strategic Intelligence Consultant",
        goal=(
            "Transform raw research and financial intelligence into a polished, "
            "professional Competitive Intelligence Brief with exactly six labeled sections. "
            "Be precise, balanced, and honest about data gaps."
        ),
        backstory=(
            "You are a former McKinsey consultant now running your own competitive "
            "intelligence practice. You write briefs that C-suite executives rely on "
            "for strategic decisions. Your writing is crisp, evidence-based, and "
            "free of fluff. You never embellish or fill gaps with assumptions — "
            "when data is missing you say so and explain why. Your briefs run "
            "400–800 words and follow a strict six-section structure."
        ),
        tools=[],    # Synthesis from context only — no external calls
        llm=get_llm(),
        verbose=True,
        max_iter=3,
    )
