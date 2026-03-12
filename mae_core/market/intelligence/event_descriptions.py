"""Event-to-text conversion functions for semantic embedding.

Converts structured market events into rich natural language descriptions
that capture the meaning, context, and significance of each event.
Quality here directly determines semantic search quality.

One job: take structured data in, return human-readable strings out.
No embedding, no storage — pure text generation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

# --- Company name lookup (top 100 most-traded tickers) ---
_COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc", "AMZN": "Amazon.com Inc", "META": "Meta Platforms Inc",
    "TSLA": "Tesla Inc", "BRK.B": "Berkshire Hathaway", "JPM": "JPMorgan Chase",
    "UNH": "UnitedHealth Group", "V": "Visa Inc", "XOM": "ExxonMobil Corporation",
    "MA": "Mastercard Inc", "LLY": "Eli Lilly and Company", "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble", "HD": "Home Depot Inc", "MRK": "Merck & Co",
    "AVGO": "Broadcom Inc", "CVX": "Chevron Corporation", "KO": "Coca-Cola Company",
    "PEP": "PepsiCo Inc", "COST": "Costco Wholesale", "ABBV": "AbbVie Inc",
    "WMT": "Walmart Inc", "BAC": "Bank of America", "MCD": "McDonald's Corporation",
    "CRM": "Salesforce Inc", "ACN": "Accenture plc", "TMO": "Thermo Fisher Scientific",
    "LIN": "Linde plc", "AMD": "Advanced Micro Devices", "TXN": "Texas Instruments",
    "NEE": "NextEra Energy", "PM": "Philip Morris International", "DHR": "Danaher Corporation",
    "RTX": "Raytheon Technologies", "HON": "Honeywell International", "AMGN": "Amgen Inc",
    "UPS": "United Parcel Service", "LOW": "Lowe's Companies", "IBM": "IBM Corporation",
    "SBUX": "Starbucks Corporation", "INTU": "Intuit Inc", "GS": "Goldman Sachs",
    "SPGI": "S&P Global Inc", "BKNG": "Booking Holdings", "ELV": "Elevance Health",
    "DE": "Deere & Company", "CAT": "Caterpillar Inc", "ADP": "ADP Inc",
    "MDLZ": "Mondelez International", "PLD": "Prologis Inc", "CB": "Chubb Limited",
    "GILD": "Gilead Sciences", "C": "Citigroup Inc", "TJX": "TJX Companies",
    "REGN": "Regeneron Pharmaceuticals", "BSX": "Boston Scientific", "MU": "Micron Technology",
    "SO": "Southern Company", "ISRG": "Intuitive Surgical", "DUK": "Duke Energy",
    "PGR": "Progressive Corporation", "BLK": "BlackRock Inc", "ZTS": "Zoetis Inc",
    "CL": "Colgate-Palmolive", "MMC": "Marsh & McLennan", "CME": "CME Group",
    "WFC": "Wells Fargo", "F": "Ford Motor Company", "GM": "General Motors",
    "GE": "GE Aerospace", "BA": "Boeing Company", "LMT": "Lockheed Martin",
    "NOC": "Northrop Grumman", "GD": "General Dynamics", "INTC": "Intel Corporation",
    "QCOM": "Qualcomm Inc", "AMAT": "Applied Materials", "LRCX": "Lam Research",
    "KLAC": "KLA Corporation", "MRVL": "Marvell Technology", "ORCL": "Oracle Corporation",
    "ADBE": "Adobe Inc", "NOW": "ServiceNow Inc", "PANW": "Palo Alto Networks",
    "CRWD": "CrowdStrike Holdings", "SNOW": "Snowflake Inc", "DDOG": "Datadog Inc",
    "COIN": "Coinbase Global", "HOOD": "Robinhood Markets",
    # ETFs
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF", "IWM": "Russell 2000 ETF",
    "GLD": "Gold ETF (SPDR)", "SLV": "Silver ETF (iShares)", "TLT": "20+ Year Treasury ETF",
    "XLE": "Energy Select Sector ETF", "XLF": "Financial Select Sector ETF",
    "XLK": "Technology Select Sector ETF", "XLV": "Health Care Select Sector ETF",
    # Futures/forex proxies
    "GC=F": "Gold Futures", "CL=F": "Crude Oil Futures (WTI)", "NQ=F": "Nasdaq Futures",
    "ES=F": "S&P 500 Futures", "EURUSD=X": "EUR/USD Forex", "GBPUSD=X": "GBP/USD Forex",
    "USDJPY=X": "USD/JPY Forex",
}


def _company(ticker: str) -> str:
    """Return 'TICKER (Company Name)' or just 'TICKER' if unknown."""
    name = _COMPANY_NAMES.get(ticker.upper(), "")
    if name:
        return f"{ticker} ({name})"
    return ticker


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def _money(value: float) -> str:
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:.2f}"


def _domain_human(domain: str) -> str:
    _map = {
        "insider": "insider trading", "macro": "macroeconomic indicators",
        "technical": "technical analysis", "events": "corporate events",
        "positioning": "futures positioning (COT)", "government": "government contract data",
        "contracts": "government contracts", "sentiment": "social sentiment",
        "fundamental": "fundamental analysis", "institutional": "institutional flows",
        "crypto": "cryptocurrency signals", "energy": "energy markets",
        "news": "news flow", "congress": "congressional trading",
        "causal": "causal cascade", "cascade": "confirmed causal chain",
    }
    return _map.get(domain, domain)


def _urgency_human(urgency: str) -> str:
    return {"immediate": "action required immediately", "hours": "action required within hours",
            "days": "action required within days"}.get(urgency, urgency)


def describe_convergence_alert(alert) -> str:
    """Convert a ConvergenceAlert to rich natural language.

    Produces a text suitable for embedding that captures direction, ticker,
    domain agreement, confidence, urgency, and causal context.
    """
    direction = alert.direction.upper()
    domains = alert.domains_converging
    n_domains = len(domains)
    confidence = getattr(alert, "confidence", 0.0)
    strength = getattr(alert, "strength", 0.0)
    urgency = getattr(alert, "urgency", "days")
    coherence = getattr(alert, "coherence", 1.0)
    sequence_score = getattr(alert, "sequence_score", 1.0)
    domain_sequence = getattr(alert, "domain_sequence", [])
    ripple_effects = getattr(alert, "ripple_effects", [])
    summary = getattr(alert, "summary", "")
    ts = getattr(alert, "timestamp", datetime.now())
    date_str = ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts)

    # Extract ticker from signals
    ticker = _extract_ticker_from_alert(alert)
    ticker_str = _company(ticker) if ticker else "market-wide"

    # Build domain description
    domain_descs = [_domain_human(d) for d in domains]
    if n_domains == 1:
        domain_text = domain_descs[0]
    elif n_domains == 2:
        domain_text = f"{domain_descs[0]} and {domain_descs[1]}"
    else:
        domain_text = ", ".join(domain_descs[:-1]) + f", and {domain_descs[-1]}"

    parts = [
        f"{direction} convergence alert for {ticker_str} detected on {date_str}.",
        f"{n_domains} independent domains are in agreement: {domain_text}.",
    ]

    # Confidence and strength
    confidence_label = "very high" if confidence > 0.80 else ("high" if confidence > 0.65
                       else ("moderate" if confidence > 0.50 else "low"))
    parts.append(
        f"Confidence: {confidence:.2f} ({confidence_label}). "
        f"Convergence strength: {strength:.2f}."
    )

    # Coherence — signal narrative consistency
    if coherence < 0.75:
        parts.append(
            f"WARNING: Signal coherence is only {coherence:.2f} — "
            f"some domains contradict each other."
        )
    else:
        parts.append(f"Signal coherence: {coherence:.2f} — domains tell a consistent story.")

    # Temporal ordering
    if domain_sequence and sequence_score != 1.0:
        seq_text = " → ".join(domain_sequence)
        if sequence_score > 1.0:
            parts.append(
                f"Domain firing sequence ({seq_text}) matches known causal lag relationships "
                f"(sequence score: {sequence_score:.2f}), boosting reliability."
            )
        else:
            parts.append(
                f"Domain firing sequence ({seq_text}) is reversed vs. known lags "
                f"(sequence score: {sequence_score:.2f}), reducing reliability."
            )

    # Urgency
    parts.append(f"Urgency: {_urgency_human(urgency)}.")

    # Causal cascade downstream effects
    if ripple_effects:
        downstream = [r.get("ticker", "") for r in ripple_effects[:3]]
        downstream_str = ", ".join(_company(t) for t in downstream if t)
        if downstream_str:
            parts.append(
                f"Causal model predicts downstream ripple effects on: {downstream_str}."
            )

    # Signal details (from summary if available)
    if summary:
        parts.append(f"System summary: {summary}")

    # Signal source detail
    signals = getattr(alert, "signals", [])
    for sig in signals[:4]:
        sig_text = _describe_signal_inline(sig)
        if sig_text:
            parts.append(sig_text)

    return " ".join(parts)


def _extract_ticker_from_alert(alert) -> str:
    """Best-effort ticker extraction from a ConvergenceAlert."""
    ticker = getattr(alert, "ticker", None)
    if ticker:
        return ticker
    for sig in getattr(alert, "signals", []):
        sym = ""
        if hasattr(sig, "metadata"):
            sym = sig.metadata.get("symbol", "")
        if not sym and hasattr(sig, "signal_id"):
            # signal_id often has format "source:TICKER:timestamp"
            parts = str(sig.signal_id).split(":")
            if len(parts) >= 2 and parts[1].isupper():
                sym = parts[1]
        if sym:
            return sym
    return ""


def _describe_signal_inline(sig) -> str:
    """One-sentence description of a Signal object for embedding context."""
    domain = getattr(sig, "domain", "")
    strength = getattr(sig, "strength", 0.0)
    direction = getattr(sig, "direction", "neutral")
    source = getattr(sig, "source", "")
    meta = getattr(sig, "metadata", {})

    if domain == "insider":
        name = meta.get("insider_name", meta.get("name", "An insider"))
        role = meta.get("role", meta.get("relationship", ""))
        value = meta.get("value", meta.get("trade_value", 0))
        val_str = _money(value) if value else ""
        role_str = f" ({role})" if role else ""
        val_str = f" worth {val_str}" if val_str else ""
        return f"Insider signal: {name}{role_str} made a {direction} trade{val_str}."

    if domain == "macro":
        indicator = meta.get("indicator", meta.get("series_id", source))
        return f"Macro signal from {indicator}: {direction} ({strength:.2f} strength)."

    if domain == "technical":
        indicator = source.replace("ta_", "").replace("_", " ").upper()
        return f"Technical signal: {indicator} shows {direction} pattern (strength {strength:.2f})."

    if domain in ("contracts", "government"):
        value = meta.get("estimated_value", meta.get("amount", 0))
        val_str = f" valued at {_money(value)}" if value else ""
        return f"Government contracts domain: {direction} signal{val_str}."

    if domain == "sentiment":
        platform = meta.get("platform", source)
        return f"Sentiment signal from {platform}: {direction} (strength {strength:.2f})."

    if domain == "institutional":
        return f"Institutional flow signal ({source}): {direction}."

    if domain == "events":
        event_type = meta.get("event_type", source)
        return f"Corporate events domain — {event_type}: {direction}."

    if domain:
        return f"{_domain_human(domain)} signal ({source}): {direction} at {strength:.2f} strength."

    return ""


def describe_market_signal(signal) -> str:
    """Convert a MarketSignal to natural language."""
    source = getattr(signal, "source", "unknown source")
    symbol = getattr(signal, "symbol", "")
    domain = getattr(signal, "domain", "")
    direction = getattr(signal, "direction", "neutral")
    strength = getattr(signal, "strength", 0.0)
    confidence = getattr(signal, "confidence", 0.5)
    asset_class = getattr(signal, "asset_class", "")
    meta = getattr(signal, "metadata", {})
    ts = getattr(signal, "timestamp", datetime.now())
    date_str = ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts)

    ticker_str = _company(symbol) if symbol else "macro market"
    domain_str = _domain_human(domain) if domain else "unclassified domain"
    asset_str = f" ({asset_class})" if asset_class and asset_class != "stock" else ""

    parts = [
        f"{direction.title()} {domain_str} signal for {ticker_str}{asset_str} "
        f"from {source} on {date_str}.",
        f"Signal strength: {strength:.2f}. Source reliability (confidence): {confidence:.2f}.",
    ]

    # Domain-specific enrichment from metadata
    if domain == "insider":
        name = meta.get("name", meta.get("insider_name", ""))
        role = meta.get("role", meta.get("relationship", ""))
        value = meta.get("value", meta.get("trade_value", 0))
        shares = meta.get("shares", 0)
        if name:
            parts.append(f"Insider: {name}" + (f" ({role})" if role else "") + ".")
        if value:
            parts.append(f"Trade value: {_money(float(value))}.")
        if shares:
            parts.append(f"Shares: {int(float(shares)):,}.")

    elif domain == "macro":
        indicator = meta.get("series_id", meta.get("indicator", source))
        value = meta.get("value", meta.get("current_value", ""))
        prev = meta.get("prev_value", meta.get("previous_value", ""))
        if indicator:
            parts.append(f"Indicator: {indicator}.")
        if value:
            change = f" (previous: {prev})" if prev else ""
            parts.append(f"Current value: {value}{change}.")

    elif domain == "technical":
        for indicator in ["rsi", "macd", "bollinger"]:
            val = meta.get(indicator, meta.get(indicator.upper(), ""))
            if val:
                parts.append(f"{indicator.upper()}: {val}.")

    elif domain in ("contracts", "government"):
        agency = meta.get("agency", meta.get("awarding_agency", ""))
        amount = meta.get("amount", meta.get("estimated_value", 0))
        if agency:
            parts.append(f"Agency: {agency}.")
        if amount:
            parts.append(f"Contract value: {_money(float(amount))}.")

    elif domain == "sentiment":
        mentions = meta.get("mentions", meta.get("mention_count", ""))
        score = meta.get("sentiment_score", meta.get("score", ""))
        if mentions:
            parts.append(f"Mention count: {mentions}.")
        if score:
            parts.append(f"Sentiment score: {score}.")

    decay_rate = getattr(signal, "decay_rate", 0.0)
    window = getattr(signal, "outcome_window_days", 14)
    parts.append(
        f"Expected outcome window: {window} days. "
        f"Signal decay rate: {decay_rate:.3f}/day."
    )

    return " ".join(parts)


def describe_pattern_template(template) -> str:
    """Convert a PatternTemplate to natural language description."""
    direction = getattr(template, "direction", "neutral")
    domain_sig = getattr(template, "domain_signature", "")
    domains = getattr(template, "domains", [])
    n_instances = getattr(template, "n_instances", 0)
    symbols_seen = getattr(template, "symbols_seen", [])
    avg_move = getattr(template, "avg_move_pct", 0.0)
    wins = getattr(template, "wins", 0)
    losses = getattr(template, "losses", 0)
    cross_validated = getattr(template, "cross_validated", False)
    confidence_mult = getattr(template, "confidence_multiplier", 1.0)
    expected_window = getattr(template, "expected_move_window_days", 14)
    lag_profile = getattr(template, "lag_profile_normalized", {})
    created_at = getattr(template, "created_at", "")

    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0

    domain_descs = [_domain_human(d) for d in domains]
    domain_text = ", ".join(domain_descs) if domain_descs else domain_sig

    parts = [
        f"{direction.title()} pattern template: {domain_sig}.",
        f"This pattern combines {domain_text}.",
        f"Observed {n_instances} times across {len(set(symbols_seen))} unique symbols.",
    ]

    if total > 0:
        wr_label = "strong" if win_rate > 0.50 else ("moderate" if win_rate > 0.33 else "weak")
        parts.append(
            f"Historical win rate: {_pct(win_rate * 100)} ({wins} wins / {losses} losses) — {wr_label} edge."
        )

    parts.append(f"Average move magnitude: {_pct(avg_move)}.")
    parts.append(f"Expected signal-to-move window: {expected_window} days.")

    if cross_validated:
        parts.append(
            f"Cross-validated across {len(set(symbols_seen))} symbols "
            f"(confidence multiplier: {confidence_mult:.2f}x) — pattern is market-wide, not ticker-specific."
        )
    else:
        parts.append("Not yet cross-validated (fewer than 3 distinct symbols).")

    # Lag profile description
    dominant_lag = max(lag_profile, key=lag_profile.get) if lag_profile else ""
    if dominant_lag:
        _lag_desc = {
            "immediate": "0-2 days before the move",
            "short": "3-5 days before the move",
            "medium": "6-10 days before the move",
            "long": "11-20 days before the move",
            "extended": "21-30 days before the move",
        }
        parts.append(
            f"Signals most commonly appear {_lag_desc.get(dominant_lag, dominant_lag)} "
            f"({_pct(lag_profile[dominant_lag] * 100)} of observations)."
        )

    if symbols_seen:
        sample = list(set(symbols_seen))[:5]
        sample_str = ", ".join(_company(s) for s in sample)
        parts.append(f"Example symbols where this pattern fired: {sample_str}.")

    if created_at:
        parts.append(f"Template first observed: {created_at[:10]}.")

    return " ".join(parts)


def describe_insider_trade(trade: dict) -> str:
    """Convert a raw insider trade dict to natural language."""
    ticker = trade.get("ticker", trade.get("symbol", "UNKNOWN"))
    name = trade.get("insider_name", trade.get("name", "Unknown insider"))
    role = trade.get("relationship", trade.get("role", trade.get("officer_title", "")))
    transaction_type = trade.get("transaction_type", trade.get("type", "Purchase"))
    shares = float(trade.get("shares", trade.get("shares_transacted", 0)) or 0)
    price = float(trade.get("price", trade.get("transaction_price", 0)) or 0)
    value = float(trade.get("value", trade.get("total_value", 0)) or 0)
    if value == 0 and shares > 0 and price > 0:
        value = shares * price
    date_str = trade.get("date", trade.get("transaction_date", trade.get("filed_at", "")))
    filing_date = trade.get("filing_date", trade.get("filed_at", ""))
    ownership_type = trade.get("ownership_type", "direct")
    shares_after = float(trade.get("shares_after", trade.get("shares_owned_following", 0)) or 0)
    source = trade.get("source", "SEC Form 4")

    ticker_str = _company(ticker)
    direction = "purchased" if "P" in str(transaction_type).upper() or "buy" in str(transaction_type).lower() else "sold"
    is_buy = direction == "purchased"

    parts = [
        f"{name}" + (f" ({role})" if role else "") +
        f" {direction} shares of {ticker_str}.",
    ]

    if shares > 0:
        parts.append(f"Transaction: {int(shares):,} shares at {_money(price)} per share.")
    if value > 0:
        parts.append(f"Total transaction value: {_money(value)}.")
    if shares_after > 0:
        parts.append(f"Total holdings after transaction: {int(shares_after):,} shares.")

    if date_str:
        parts.append(f"Transaction date: {date_str}.")
    if filing_date and filing_date != date_str:
        parts.append(f"Filed with SEC: {filing_date}.")

    # Conviction signals
    if is_buy:
        if value > 1_000_000:
            parts.append("This is a large conviction purchase exceeding $1M.")
        if "ceo" in str(role).lower() or "cfo" in str(role).lower():
            parts.append("C-suite executive purchase — highest conviction insider signal.")
        elif "director" in str(role).lower():
            parts.append("Board director purchase — strong insider knowledge signal.")
    else:
        parts.append(
            "Insider sale — may be planned liquidation (10b5-1 plan) or genuine distribution."
        )

    parts.append(f"Ownership type: {ownership_type}. Data source: {source}.")

    return " ".join(parts)


def describe_economic_event(event: dict) -> str:
    """Convert an economic calendar event or FRED data point to natural language."""
    event_type = event.get("event", event.get("name", event.get("title", "Economic event")))
    date_str = event.get("date", event.get("release_date", ""))
    actual = event.get("actual", event.get("value", ""))
    forecast = event.get("forecast", event.get("estimate", ""))
    previous = event.get("previous", event.get("prev_value", ""))
    country = event.get("country", "US")
    impact = event.get("impact", event.get("importance", "medium"))
    series_id = event.get("series_id", "")

    parts = [f"Economic event: {event_type}"]
    if date_str:
        parts[0] += f" on {date_str}"
    if country:
        parts[0] += f" ({country})."

    if actual:
        surprise = ""
        if forecast:
            try:
                a, f_val = float(str(actual).replace("%", "")), float(str(forecast).replace("%", ""))
                diff = a - f_val
                if abs(diff) > 0.1:
                    surprise = f" — surprise of {'+' if diff > 0 else ''}{diff:.2f} vs forecast"
            except (ValueError, TypeError):
                pass
        parts.append(f"Actual: {actual}{surprise}.")
        if forecast:
            parts.append(f"Forecast was: {forecast}.")
        if previous:
            parts.append(f"Previous: {previous}.")

    if series_id:
        parts.append(f"FRED series: {series_id}.")

    impact_str = {"high": "high-impact", "medium": "medium-impact", "low": "low-impact"}.get(
        str(impact).lower(), str(impact)
    )
    parts.append(f"Market impact classification: {impact_str}.")

    return " ".join(parts)


def describe_congressional_trade(trade: dict) -> str:
    """Convert a congressional stock trade to natural language."""
    member = trade.get("representative", trade.get("member", trade.get("senator", "Unknown member")))
    chamber = trade.get("chamber", "Congress")
    ticker = trade.get("ticker", trade.get("symbol", ""))
    asset_name = trade.get("asset_name", trade.get("asset_description", ""))
    transaction_type = trade.get("type", trade.get("transaction_type", "Purchase"))
    amount = trade.get("amount", trade.get("amount_range", ""))
    date_str = trade.get("transaction_date", trade.get("date", ""))
    committee = trade.get("committee", "")
    state = trade.get("state", "")

    ticker_str = _company(ticker) if ticker else asset_name or "an unspecified asset"
    direction = "purchased" if "purchase" in str(transaction_type).lower() else "sold"

    parts = [
        f"{member}" + (f" ({chamber}" + (f", {state}" if state else "") + ")" if chamber else "") +
        f" {direction} {ticker_str}.",
    ]

    if amount:
        parts.append(f"Reported transaction range: {amount}.")
    if date_str:
        parts.append(f"Transaction date: {date_str}.")
    if committee:
        parts.append(
            f"Committee membership: {committee}. "
            f"This creates potential information advantage if the committee oversees related legislation."
        )

    parts.append(
        "Congressional trades are filed under the STOCK Act and represent privileged "
        "access to legislative and regulatory intelligence."
    )

    return " ".join(parts)


def describe_contract_award(contract: dict) -> str:
    """Convert a government contract award to natural language."""
    company = contract.get("recipient_name", contract.get("company", "Unknown company"))
    agency = contract.get("awarding_agency", contract.get("agency", "Unknown agency"))
    amount = float(contract.get("amount", contract.get("base_and_all_options_value", 0)) or 0)
    description = contract.get("description", contract.get("contract_description", ""))
    date_str = contract.get("period_of_performance_start", contract.get("award_date", contract.get("date", "")))
    ticker = contract.get("ticker", "")
    pop_end = contract.get("period_of_performance_end", "")

    parts = [
        f"Government contract awarded to {company}" +
        (f" ({_company(ticker)})" if ticker else "") +
        f" by {agency}.",
    ]

    if amount > 0:
        parts.append(f"Contract value: {_money(amount)}.")
    if description:
        parts.append(f"Scope: {description[:200]}.")
    if date_str:
        parts.append(f"Performance start: {date_str}.")
    if pop_end:
        parts.append(f"Performance end: {pop_end}.")

    parts.append(
        "Government contracts provide visible, durable revenue and often signal "
        "sector prioritization by federal agencies."
    )

    return " ".join(parts)
