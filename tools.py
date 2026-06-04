"""
Custom tools for the Multi-Agent Competitive Intelligence System.

- DuckDuckGoTool: Free web search via DuckDuckGo (no API key needed)
- WikipediaTool: Wikipedia article fetcher
- YFinanceTool: Yahoo Finance financial data for public companies
"""

import json
import logging
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

logger = logging.getLogger(__name__)


# ── Input schemas ──────────────────────────────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(description="The search query string to look up on the web.")


class WikiInput(BaseModel):
    query: str = Field(description="The company or topic name to search on Wikipedia.")


class TickerInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol (e.g. TSLA, NVDA, AAPL).")


# ── Tool implementations ───────────────────────────────────────────────────────

class DuckDuckGoTool(BaseTool):
    """Web search using DuckDuckGo — free, no API key required."""

    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Search the internet for current information about a company, news, "
        "competitors, or any topic. Input must be a concise search query."
    )
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        from ddgs import DDGS
        logger.info(f"[DuckDuckGo] Searching: {query!r}")
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=6))
        except Exception as exc:
            raise RuntimeError(f"DuckDuckGo search failed: {exc}") from exc

        if not hits:
            raise ValueError(f"DuckDuckGo returned no results for: {query!r}")

        lines = []
        for h in hits:
            lines.append(f"Title: {h.get('title', 'N/A')}")
            lines.append(f"Snippet: {h.get('body', 'N/A')}")
            lines.append(f"URL: {h.get('href', 'N/A')}")
            lines.append("---")
        return "\n".join(lines)


class WikipediaTool(BaseTool):
    """Fetch a Wikipedia article summary (first 3 000 chars)."""

    name: str = "Wikipedia Lookup"
    description: str = (
        "Retrieve background information about a company or topic from Wikipedia. "
        "Input should be the company name or a specific topic."
    )
    args_schema: type[BaseModel] = WikiInput

    def _run(self, query: str) -> str:
        import wikipedia as wiki_lib
        logger.info(f"[Wikipedia] Fetching: {query!r}")

        def fetch(q: str, suggest: bool) -> str:
            page = wiki_lib.page(q, auto_suggest=suggest)
            return f"**{page.title}**\n\n{page.content[:3000]}"

        # Strategy: try exact match first (auto_suggest=False), then broaden
        attempts = [
            (query, False),
            (f"{query} company", False),
            (query, True),
        ]
        for q, suggest in attempts:
            try:
                return fetch(q, suggest)
            except wiki_lib.exceptions.DisambiguationError as e:
                # Among disambiguation options prefer ones containing the query keyword
                keyword = query.lower()
                preferred = [opt for opt in e.options if keyword in opt.lower()]
                candidates = preferred if preferred else e.options
                for option in candidates[:3]:
                    try:
                        logger.warning(f"[Wikipedia] Disambiguation — trying option: {option!r}")
                        return fetch(option, False)
                    except Exception:
                        continue
            except wiki_lib.exceptions.PageError:
                continue
            except Exception as exc:
                logger.warning(f"[Wikipedia] Error on {q!r}: {exc}")
                continue

        raise ValueError(f"Wikipedia could not find a page for: {query!r}")


class YFinanceTool(BaseTool):
    """Retrieve financial metrics for a publicly traded company via Yahoo Finance."""

    name: str = "Yahoo Finance Lookup"
    description: str = (
        "Get real financial data for a publicly traded company using its stock ticker symbol. "
        "Returns market cap, revenue, margins, P/E ratio, and a business summary. "
        "Input MUST be the ticker symbol (e.g. NVDA, TSLA, AAPL), NOT the company name."
    )
    args_schema: type[BaseModel] = TickerInput

    def _run(self, ticker: str) -> str:
        import yfinance as yf
        ticker = ticker.strip().upper()
        logger.info(f"[yFinance] Fetching ticker: {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.info

        # Detect invalid / no-data tickers
        if not info or info.get("quoteType") is None:
            raise ValueError(f"No data found for ticker '{ticker}'. Verify the symbol is correct.")

        def fmt_money(val):
            if val is None:
                return "N/A"
            if val >= 1e12:
                return f"${val/1e12:.2f}T"
            if val >= 1e9:
                return f"${val/1e9:.2f}B"
            if val >= 1e6:
                return f"${val/1e6:.2f}M"
            return f"${val:,.0f}"

        def fmt_pct(val):
            return f"{val*100:.1f}%" if val is not None else "N/A"

        data = {
            "Company Name": info.get("longName", "N/A"),
            "Ticker": ticker,
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Market Cap": fmt_money(info.get("marketCap")),
            "Revenue (TTM)": fmt_money(info.get("totalRevenue")),
            "Net Income (TTM)": fmt_money(info.get("netIncomeToCommon")),
            "Gross Margin": fmt_pct(info.get("grossMargins")),
            "Operating Margin": fmt_pct(info.get("operatingMargins")),
            "Revenue Growth YoY": fmt_pct(info.get("revenueGrowth")),
            "P/E Ratio (Trailing)": info.get("trailingPE", "N/A"),
            "EPS (TTM)": info.get("trailingEps", "N/A"),
            "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Full-Time Employees": info.get("fullTimeEmployees", "N/A"),
            "Business Summary": (info.get("longBusinessSummary") or "N/A")[:600],
        }

        return json.dumps(data, indent=2)
