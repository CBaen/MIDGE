"""Plain-language alert formatter — zero financial jargon.

Guiding Light's directive: "I don't even know what bullish means."

This module translates MIDGE's internal financial vocabulary into
clear, actionable English that anyone can understand. Five sections:
WHAT, HISTORY, TIMING, ACTION, TRACKING.

All translation maps live here. One file, one responsibility.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Direction Translation ──────────────────────────────────────────

DIRECTION_PLAIN = {
    "bullish": "expected to increase in price",
    "bearish": "expected to decrease in price",
    "up": "expected to increase in price",
    "down": "expected to decrease in price",
    "buy": "increase",
    "sell": "decrease",
}

# ── Domain Translation ─────────────────────────────────────────────

DOMAIN_PLAIN = {
    "insider": "company executives and officers bought or sold shares",
    "macro": "economic conditions shifted in a relevant direction",
    "technical": "price chart patterns formed that historically precede moves",
    "events": "the company filed significant disclosures with regulators",
    "positioning": "large professional traders shifted their positions",
    "government": "elected officials made related stock trades",
    "congress": "elected officials made related stock trades",
    "contracts": "government contracts were awarded in this sector",
    "sentiment": "public investor discussion spiked on social media",
    "fundamental": "company financial data showed notable changes",
    "fundamentals": "company financial data showed notable changes",
    "institutional": "large institutional investors changed their holdings",
    "crypto": "cryptocurrency market activity showed correlated movement",
    "price": "price movement patterns matched historical precedents",
}

# ── Source Translation ─────────────────────────────────────────────

SOURCE_PLAIN = {
    "sec_form4": "SEC insider trade filing",
    "sec_form8k": "SEC material event disclosure",
    "sec_efts": "SEC electronic filing",
    "fred_macro": "Federal Reserve economic data",
    "cot_positioning": "futures market positioning data",
    "ta_rsi": "momentum indicator",
    "ta_macd": "trend indicator",
    "ta_bollinger": "volatility indicator",
    "finviz_unusual_volume": "unusual trading volume",
    "finnhub_earnings": "earnings report data",
    "congressional": "congressional stock trade",
    "insider_cluster": "multiple insiders trading together",
    "contract": "government contract award",
    "contract_prediction": "government contract prediction",
    "hiring_tracker": "company hiring activity",
    "sam_gov": "federal contracting opportunity",
    "correlation": "cross-market correlation",
    "convergence_combo": "multiple independent signals aligning",
    "session_sweep": "large institutional order",
    "session_sweep_ifvg": "institutional order with price gap",
    "stocktwits": "social media discussion",
    "vix_client": "market fear indicator",
    "google_trends": "public search interest",
}

# ── Tier Translation ───────────────────────────────────────────────

TIER_PLAIN = {
    "high": "strong",
    "medium": "moderate",
    "low": "emerging",
}

# ── Company Descriptions ───────────────────────────────────────────
# Top traded tickers. Fallback: "[TICKER], a publicly traded company"

COMPANY_DESCRIPTIONS = {
    "AAPL": "Apple, maker of iPhones and Mac computers",
    "MSFT": "Microsoft, the Windows and cloud computing company",
    "AMZN": "Amazon, the e-commerce and cloud services company",
    "NVDA": "NVIDIA, a semiconductor company that makes graphics chips",
    "GOOGL": "Google (Alphabet), the search and advertising company",
    "GOOG": "Google (Alphabet), the search and advertising company",
    "META": "Meta (Facebook), the social media company",
    "TSLA": "Tesla, the electric vehicle manufacturer",
    "BRK.B": "Berkshire Hathaway, Warren Buffett's investment company",
    "UNH": "UnitedHealth, a healthcare and insurance company",
    "JPM": "JPMorgan Chase, one of the largest banks",
    "V": "Visa, the payment processing company",
    "JNJ": "Johnson & Johnson, a healthcare products company",
    "XOM": "ExxonMobil, an oil and gas company",
    "MA": "Mastercard, the payment processing company",
    "PG": "Procter & Gamble, a consumer goods company",
    "HD": "Home Depot, the home improvement retailer",
    "CVX": "Chevron, an energy company",
    "MRK": "Merck, a pharmaceutical company",
    "ABBV": "AbbVie, a pharmaceutical company",
    "LLY": "Eli Lilly, a pharmaceutical company",
    "AVGO": "Broadcom, a semiconductor company",
    "PEP": "PepsiCo, the food and beverage company",
    "KO": "Coca-Cola, the beverage company",
    "COST": "Costco, the wholesale retailer",
    "TMO": "Thermo Fisher Scientific, a life sciences company",
    "MCD": "McDonald's, the fast food restaurant chain",
    "WMT": "Walmart, the retail chain",
    "CSCO": "Cisco, a networking equipment company",
    "ACN": "Accenture, a consulting and technology services company",
    "ABT": "Abbott Laboratories, a medical devices company",
    "CRM": "Salesforce, the cloud software company",
    "DHR": "Danaher, a life sciences and diagnostics company",
    "AMD": "AMD, a semiconductor company",
    "INTC": "Intel, a semiconductor company",
    "NFLX": "Netflix, the streaming entertainment company",
    "DIS": "Disney, the entertainment and media company",
    "NKE": "Nike, the athletic apparel company",
    "BA": "Boeing, the aerospace and defense company",
    "GS": "Goldman Sachs, an investment bank",
    "PYPL": "PayPal, a digital payments company",
    "SQ": "Block (Square), a financial technology company",
    "SNAP": "Snap, the social media company behind Snapchat",
    "UBER": "Uber, the ride-sharing and delivery company",
    "COIN": "Coinbase, a cryptocurrency exchange",
    "PLTR": "Palantir, a data analytics company",
    "SOFI": "SoFi, a financial technology company",
    "RIVN": "Rivian, an electric vehicle manufacturer",
    "SPY": "S&P 500 index fund (tracks the 500 largest US companies)",
    "QQQ": "Nasdaq 100 index fund (tracks major tech companies)",
    "IWM": "Russell 2000 index fund (tracks small companies)",
    "DIA": "Dow Jones index fund (tracks 30 major companies)",
    "GLD": "Gold ETF (tracks the price of gold)",
    "SLV": "Silver ETF (tracks the price of silver)",
    "USO": "Oil ETF (tracks the price of crude oil)",
    "TLT": "Treasury Bond ETF (tracks long-term US government bonds)",
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
}


def _describe_company(ticker: str) -> str:
    """Return a plain description of a company, or a generic fallback."""
    if ticker in COMPANY_DESCRIPTIONS:
        return f"{ticker} ({COMPANY_DESCRIPTIONS[ticker]})"
    return f"{ticker}, a publicly traded asset"


def _translate_direction(direction: str) -> str:
    """Convert financial direction to plain English."""
    return DIRECTION_PLAIN.get(direction, direction)


def _translate_domains(domains: list[str]) -> list[str]:
    """Convert domain codes to plain descriptions."""
    return [DOMAIN_PLAIN.get(d, d) for d in domains]


def _translate_sources(sources: list[str]) -> list[str]:
    """Convert source codes to plain descriptions."""
    return [SOURCE_PLAIN.get(s, s) for s in sources]


# ── Alert Sections ─────────────────────────────────────────────────


def _section_what(ticker: str, direction: str, domains: list[str],
                  n_patterns: int = 0) -> str:
    """WHAT section: what MIDGE found."""
    company = _describe_company(ticker)
    dir_text = _translate_direction(direction)
    domain_descs = _translate_domains(domains)

    signals_text = "; ".join(domain_descs)
    if n_patterns > 1:
        intro = f"MIDGE detected {n_patterns} independent patterns aligning on {company}."
    else:
        intro = f"MIDGE detected a pattern on {company}."
    return (
        f"WHAT: {intro} Multiple independent signals aligned: "
        f"{signals_text}. The price is {dir_text}."
    )


def _section_history(n_instances: int, n_symbols: int,
                     win_rate: Optional[float], sample_size: int) -> str:
    """HISTORY section: how often this pattern played out before."""
    if n_instances == 0 and sample_size == 0:
        return "HISTORY: This is a new pattern without enough historical data to assess yet."
    parts = []
    if n_instances > 0:
        parts.append(
            f"This same combination appeared {n_instances} times in the past"
        )
        if n_symbols > 1:
            parts[-1] += f" across {n_symbols} different companies"
        parts[-1] += "."
    if win_rate is not None and sample_size >= 5:
        pct = round(win_rate * 100)
        parts.append(
            f"In {pct}% of those cases ({sample_size} graded), "
            f"the expected price move followed."
        )
    elif sample_size > 0:
        parts.append(
            f"Only {sample_size} cases have been graded so far — "
            f"not enough to calculate a reliable rate."
        )
    return "HISTORY: " + " ".join(parts)


def _section_timing(window_days: int, window_source: str = "dynamic") -> str:
    """TIMING section: when the move is expected."""
    if window_source == "dynamic":
        return (
            f"TIMING: Based on historical patterns, if this move occurs, "
            f"it typically happens within {window_days} days."
        )
    return (
        f"TIMING: MIDGE will monitor this for {window_days} days. "
        f"There isn't enough timing data yet to narrow the window."
    )


def _section_action(tier: str, confidence: float) -> str:
    """ACTION section: concrete guidance."""
    tier_text = TIER_PLAIN.get(tier, tier)
    conf_pct = round(confidence * 100)
    return (
        f"ACTION: This is a {tier_text} signal ({conf_pct}% confidence). "
        f"MIDGE does not make financial decisions — this is information for "
        f"you to evaluate. If you choose to act on this, do your own research first."
    )


def _section_tracking(ticker: str) -> str:
    """TRACKING section: what happens next."""
    return f"TRACKING: MIDGE will monitor {ticker} and report back with updates."


# ── Main Formatter ─────────────────────────────────────────────────


def format_pattern_stack_alert(
    stack,
    window_days: int = 14,
    window_source: str = "fallback",
) -> str:
    """Format a PatternStack as a complete plain-language alert.

    Args:
        stack: PatternStack from pattern_watcher
        window_days: expected outcome window
        window_source: "dynamic" or "fallback"

    Returns:
        Multi-line plain-English alert with 5 sections.
    """
    # Gather data from the stack
    ticker = stack.symbol
    direction = stack.direction
    domains = set()
    total_instances = 0
    total_symbols = set()
    total_wins = 0
    total_losses = 0

    for act in stack.activations:
        tmpl = act.template
        domains.update(getattr(tmpl, "domains", []))
        total_instances += getattr(tmpl, "n_instances", 0)
        total_symbols.update(getattr(tmpl, "symbols_seen", []))
        total_wins += getattr(tmpl, "wins", 0)
        total_losses += getattr(tmpl, "losses", 0)

    sample_size = total_wins + total_losses
    win_rate = total_wins / sample_size if sample_size > 0 else None

    sections = [
        _section_what(ticker, direction, sorted(domains), len(stack.activations)),
        _section_history(total_instances, len(total_symbols), win_rate, sample_size),
        _section_timing(window_days, window_source),
        _section_action(stack.tier, stack.stack_confidence),
        _section_tracking(ticker),
    ]
    return "\n\n".join(sections)


def format_convergence_alert(
    alert,
    ticker: str,
    window_days: int = 14,
) -> str:
    """Format a ConvergenceAlert as a plain-language alert.

    Args:
        alert: ConvergenceAlert from convergence_alerter
        ticker: primary symbol
        window_days: outcome window

    Returns:
        Multi-line plain-English alert.
    """
    direction = getattr(alert, "direction", "neutral")
    domains = getattr(alert, "domains_converging", [])
    confidence = float(getattr(alert, "confidence", 0.0))

    sections = [
        _section_what(ticker, direction, domains),
        f"HISTORY: This convergence combines {len(domains)} independent signal domains.",
        _section_timing(window_days, "dynamic" if len(domains) >= 3 else "fallback"),
        _section_action("medium", confidence),
        _section_tracking(ticker),
    ]
    return "\n\n".join(sections)


# ── JSONL Writer ───────────────────────────────────────────────────

ALERTS_PATH = Path("data/midge/alerts_human.jsonl")


def write_plain_alert(message: str, ticker: str, direction: str,
                      source: str = "pattern_stack",
                      metadata: Optional[dict] = None) -> None:
    """Append a plain-language alert to alerts_human.jsonl."""
    try:
        ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "direction": direction,
            "source": source,
            "message": message,
        }
        if metadata:
            record["metadata"] = metadata
        with open(ALERTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        logger.debug("Failed to write plain-language alert", exc_info=True)
