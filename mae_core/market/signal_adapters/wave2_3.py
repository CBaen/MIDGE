"""Wave 2+3 signal adapters — crypto, OpenInsider, 13F, FinViz, Finnhub real-time, suppression.

Converts Wave 2 (real-time + crypto) and Wave 3 (data enrichment) sources
into normalized MarketSignal objects.

Sub-modules:
  wave2_3_insider.py  — from_openinsider, from_13f_holding, from_13f_filer_activity,
                        from_activist_filing, from_finviz_insider
  wave2_3_technical.py — from_finviz_unusual_volume, from_finviz_short_squeeze,
                         from_finnhub_realtime, from_suppression_event,
                         from_economic_surprise, from_massive_snapshot, from_kalshi_mover
"""

from __future__ import annotations

from datetime import datetime

from mae_core.market.signal import MarketSignal, _ensure_datetime

# Re-export insider adapters
from mae_core.market.signal_adapters.wave2_3_insider import (
    from_openinsider,
    from_13f_holding,
    from_13f_filer_activity,
    from_activist_filing,
    from_finviz_insider,
)

# Re-export technical/market adapters
from mae_core.market.signal_adapters.wave2_3_technical import (
    from_finviz_unusual_volume,
    from_finviz_short_squeeze,
    from_finnhub_realtime,
    from_suppression_event,
    from_economic_surprise,
    from_binance_funding,
    from_kalshi_mover,
    from_massive_snapshot,
)

__all__ = [
    "from_crypto_signal",
    "from_openinsider",
    "from_13f_holding",
    "from_13f_filer_activity",
    "from_activist_filing",
    "from_finviz_insider",
    "from_finviz_unusual_volume",
    "from_finviz_short_squeeze",
    "from_finnhub_realtime",
    "from_suppression_event",
    "from_economic_surprise",
    "from_binance_funding",
    "from_kalshi_mover",
    "from_massive_snapshot",
]


def from_crypto_signal(crypto_price) -> MarketSignal:
    """Convert a CryptoPrice (CoinGecko) or CryptoAsset (CoinCap) to a MarketSignal.

    Both sources expose a `change_24h_pct` field and `price_usd`. The adapter
    accepts either via duck-typing — CoinGecko objects have `coin_id`, CoinCap
    objects have `asset_id`. Direction gates at ±3% (crypto noise floor).
    Strength scales linearly to 10% (full-strength move).
    """
    change_24h = getattr(crypto_price, "change_24h_pct", 0.0) or 0.0
    symbol = getattr(crypto_price, "symbol", "") or ""
    price_usd = getattr(crypto_price, "price_usd", 0.0) or 0.0
    volume = (
        getattr(crypto_price, "volume_24h", None)
        or getattr(crypto_price, "volume_24h_usd", None)
        or 0.0
    )
    market_cap = (
        getattr(crypto_price, "market_cap", None)
        or getattr(crypto_price, "market_cap_usd", None)
        or 0.0
    )

    # Detect source by distinguishing attribute
    if hasattr(crypto_price, "coin_id"):
        source = "crypto_coingecko"
        raw_id = getattr(crypto_price, "coin_id", "")
        raw_type = "CryptoPrice"
        last_updated = getattr(crypto_price, "last_updated", "")
        event_dt = _ensure_datetime(last_updated) if last_updated else datetime.now()
        extra_meta = {
            "coin_id": raw_id,
            "change_7d_pct": getattr(crypto_price, "change_7d_pct", 0.0),
        }
    else:
        source = "crypto_coincap"
        raw_id = getattr(crypto_price, "asset_id", "")
        raw_type = "CryptoAsset"
        event_dt = datetime.now()
        extra_meta = {
            "asset_id": raw_id,
            "rank": getattr(crypto_price, "rank", 0),
            "vwap_24h": getattr(crypto_price, "vwap_24h", None),
        }

    if change_24h > 3.0:
        direction = "bullish"
    elif change_24h < -3.0:
        direction = "bearish"
    else:
        direction = "neutral"

    strength = min(1.0, abs(change_24h) / 10.0)

    outcome_symbol = symbol.upper() + "-USD"
    signal_id = f"crypto:{source}:{symbol}:{event_dt.date()}"

    return MarketSignal(
        signal_id=signal_id,
        source=source,
        symbol=symbol.upper(),
        asset_class="crypto",
        domain="crypto",
        direction=direction,
        strength=strength,
        confidence=0.50,
        decay_rate=0.50,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=outcome_symbol,
        outcome_window_days=3,
        raw_id=raw_id,
        raw_type=raw_type,
        metadata={
            "symbol": symbol.upper(),
            "price_usd": price_usd,
            "volume_24h": volume,
            "change_24h_pct": change_24h,
            "market_cap": market_cap,
            **extra_meta,
        },
    )
