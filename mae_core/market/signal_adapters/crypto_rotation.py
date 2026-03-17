"""Crypto rotation signal adapters — BTC dominance and derivatives positioning.

Converts CoinGecko GlobalCryptoData (BTC dominance) and BinanceDerivativesClient
CrowdedTradeScore objects into normalized MarketSignal objects.

BTC dominance thresholds:
  > 60%   → alts suppressed (money concentrated in BTC) → "bearish" for alts
  < 55%   → alt season starting (capital rotating out of BTC) → "bullish" for alts
  falling rapidly (>2% in 7 days) → capital leaving BTC → "bullish" for alts

Why BTC dominance matters for convergence:
  BTC.D is an INDEPENDENT structural signal — not a price signal, not a volume
  signal, but a market structure / capital flow signal. When combined with
  crypto price momentum and derivatives positioning, it creates a third domain
  (positioning/structural) that satisfies Law 2 (No Bare Dyads).

  BTC.D falling + alt price rising + normal funding rates = 3-domain convergence
  that has historically been the most reliable alt season entry.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from mae_core.market.signal import MarketSignal

logger = logging.getLogger("midge.market.signal_adapters.crypto_rotation")

# BTC dominance thresholds (percentage points, e.g. 60.0 = 60%)
BTC_DOM_HIGH = 60.0          # Above this → alts suppressed (bearish for alts)
BTC_DOM_LOW = 55.0           # Below this → alt season possible (bullish for alts)
BTC_DOM_RAPID_FALL = 2.0     # 7-day decline of >2pp → capital rotation in progress
BTC_DOM_RAPID_RISE = 2.0     # 7-day rise of >2pp → risk-off, flight to BTC quality


def from_btc_dominance(
    global_data,
    prior_dominance: Optional[float] = None,
    days_window: int = 7,
) -> Optional[MarketSignal]:
    """Convert a CoinGecko GlobalCryptoData to a crypto rotation MarketSignal.

    Args:
        global_data: GlobalCryptoData with btc_dominance field (float, %)
        prior_dominance: BTC.D reading from `days_window` ago (for velocity)
        days_window: How many days ago the prior reading was taken

    Returns:
        MarketSignal with domain="crypto_structure" or None if no signal.
        Returns None for neutral readings (dominance between 55-60%
        with no strong velocity component) to avoid signal noise.
    """
    if global_data is None:
        return None

    btc_dom = getattr(global_data, "btc_dominance", None)
    if btc_dom is None:
        return None

    btc_dom = float(btc_dom)
    direction = "neutral"
    strength = 0.0
    signal_subtype = "level"
    metadata_extra: dict = {}

    # --- Threshold-based level signals ---
    if btc_dom > BTC_DOM_HIGH:
        # High BTC dominance = capital concentrated in BTC = bearish for alts
        direction = "bearish"
        # Strength scales from 60% baseline to 70% (full strength)
        strength = min(1.0, (btc_dom - BTC_DOM_HIGH) / 10.0)
        signal_subtype = "btc_dominance_high"
        metadata_extra["note"] = f"BTC.D={btc_dom:.1f}%: capital concentrated in BTC, alts suppressed"

    elif btc_dom < BTC_DOM_LOW:
        # Low BTC dominance = capital rotating into alts = bullish for alts
        direction = "bullish"
        # Strength scales from 55% baseline down to 45% (full strength)
        strength = min(1.0, (BTC_DOM_LOW - btc_dom) / 10.0)
        signal_subtype = "btc_dominance_low"
        metadata_extra["note"] = f"BTC.D={btc_dom:.1f}%: alt season territory, capital rotating to alts"

    # --- Velocity signal (overrides level signal if strong enough) ---
    if prior_dominance is not None:
        dom_change = btc_dom - float(prior_dominance)
        metadata_extra["dom_change_7d"] = round(dom_change, 3)

        if dom_change < -BTC_DOM_RAPID_FALL:
            # BTC.D falling rapidly → capital leaving BTC → bullish for alts
            velocity_direction = "bullish"
            velocity_strength = min(1.0, abs(dom_change) / 5.0)
            signal_subtype = "btc_dominance_falling"
            metadata_extra["note"] = (
                f"BTC.D falling {abs(dom_change):.1f}pp in {days_window}d "
                f"({prior_dominance:.1f}% → {btc_dom:.1f}%): alt season rotation in progress"
            )
            # Use velocity signal if stronger or if direction changes
            if velocity_strength > strength or direction == "neutral":
                direction = velocity_direction
                strength = velocity_strength

        elif dom_change > BTC_DOM_RAPID_RISE:
            # BTC.D rising rapidly → risk-off flight to BTC → bearish for alts
            velocity_direction = "bearish"
            velocity_strength = min(1.0, abs(dom_change) / 5.0)
            signal_subtype = "btc_dominance_rising"
            metadata_extra["note"] = (
                f"BTC.D rising {dom_change:.1f}pp in {days_window}d "
                f"({prior_dominance:.1f}% → {btc_dom:.1f}%): risk-off, capital moving to BTC"
            )
            if velocity_strength > strength or direction == "neutral":
                direction = velocity_direction
                strength = velocity_strength

    if direction == "neutral" or strength < 0.05:
        return None  # No significant signal

    signal_id = f"btc_dominance:{signal_subtype}:{datetime.now().date()}"

    return MarketSignal(
        signal_id=signal_id,
        source="crypto_rotation",
        symbol="BTC",
        asset_class="crypto",
        domain="crypto_structure",
        direction=direction,
        strength=round(strength, 3),
        confidence=0.60,
        decay_rate=0.15,
        timestamp=datetime.now(),
        received_at=datetime.now(),
        outcome_symbol="ETH-USD",
        outcome_window_days=14,
        raw_id="",
        raw_type="GlobalCryptoData",
        metadata={
            "btc_dominance": btc_dom,
            "prior_dominance": prior_dominance,
            "days_window": days_window,
            "signal_subtype": signal_subtype,
            "total_market_cap_usd": getattr(global_data, "total_market_cap_usd", None),
            "active_cryptocurrencies": getattr(global_data, "active_cryptocurrencies", None),
            **metadata_extra,
        },
    )


def from_crowded_trade_score(score) -> Optional[MarketSignal]:
    """Convert a BinanceDerivativesClient CrowdedTradeScore to a MarketSignal.

    CrowdedTradeScore synthesizes funding rate + OI divergence + long/short ratio.
    Only fires when score >= 2 (at least 2 of 3 conditions satisfied).

    Domain: "positioning" — same as binance_funding, so they share a domain.
    This is intentional: both are derivatives positioning signals. Combined
    with crypto price (domain="crypto") and BTC structure (domain="crypto_structure"),
    a three-domain convergence is possible.

    Args:
        score: CrowdedTradeScore dataclass from BinanceDerivativesClient

    Returns:
        MarketSignal or None if score < 2 (not actionable).
    """
    if score is None:
        return None

    crowd_score = getattr(score, "score", 0)
    if crowd_score < 2:
        return None   # Below actionability threshold — filter noise

    symbol_raw = getattr(score, "symbol", "") or "BTCUSDT"
    clean_symbol = symbol_raw.replace("USDT", "").replace("BUSD", "").replace("PERP", "")
    direction = getattr(score, "direction", "neutral")
    if direction == "neutral":
        return None

    # Strength: 0.5 for 2-of-3, 1.0 for 3-of-3
    strength = 0.5 + (crowd_score - 2) * 0.5

    ts = getattr(score, "timestamp", None)
    if not isinstance(ts, datetime):
        ts = datetime.now()

    signal_id = f"crowded_trade:{symbol_raw}:{ts.strftime('%Y%m%dT%H%M')}"

    conditions_met = []
    if getattr(score, "funding_extreme", False):
        conditions_met.append("funding_extreme")
    if getattr(score, "oi_diverging", False):
        conditions_met.append("oi_diverging")
    if getattr(score, "long_extreme", False):
        conditions_met.append("long_short_extreme")

    return MarketSignal(
        signal_id=signal_id,
        source="binance_derivatives",
        symbol=clean_symbol,
        asset_class="crypto",
        domain="positioning",
        direction=direction,
        strength=round(strength, 3),
        confidence=0.65,          # High confidence — mechanical positioning signal
        decay_rate=0.60,          # Liquidation cascades happen fast — short half-life
        timestamp=ts,
        received_at=datetime.now(),
        outcome_symbol=clean_symbol + "-USD",
        outcome_window_days=3,    # Cascades play out within 1-3 days
        raw_id=symbol_raw,
        raw_type="CrowdedTradeScore",
        metadata={
            "symbol": symbol_raw,
            "crowded_score": crowd_score,
            "conditions_met": conditions_met,
            "funding_extreme": getattr(score, "funding_extreme", False),
            "oi_diverging": getattr(score, "oi_diverging", False),
            "long_extreme": getattr(score, "long_extreme", False),
            "funding_rate": getattr(score, "funding_rate", None),
            "oi_change_pct": getattr(score, "oi_change_pct", None),
            "long_short_ratio": getattr(score, "long_short_ratio", None),
        },
    )
