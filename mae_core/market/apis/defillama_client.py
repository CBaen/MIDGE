"""
defillama_client.py - DefiLlama DeFi Capital Flow Client

DefiLlama tracks $200B+ in DeFi (Decentralized Finance) capital across 50+
blockchains. TVL (Total Value Locked) is the DeFi equivalent of market cap —
money that has been deployed into on-chain protocols.

Why TVL matters for MIDGE:
  TVL DROP >10%: Capital flight. Funds exiting DeFi = risk-off signal, often
    precedes crypto price drops. When smart money leaves first, it shows up
    in TVL before price.
  TVL SURGE >10%: Capital inflow. New money entering DeFi = risk-on signal,
    often precedes or confirms crypto price appreciation.
  STABLECOIN SUPPLY GROWTH: New money entering crypto ecosystem (USDC/USDT
    minted when fiat converts to crypto). Bullish for crypto broadly.
  STABLECOIN SUPPLY CONTRACTION: Redemptions — fiat leaving ecosystem. Bearish.
  YIELD SPIKE: Protocols offering unusual yields attract capital or signal
    stress (high yield = high risk or supply/demand imbalance).

API: https://api.llama.fi — completely free, no auth, no rate limits specified.
     Being conservative at 2s/request.
Stablecoins: https://stablecoins.llama.fi
Yields: https://yields.llama.fi

No API key needed. No registration required.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

LLAMA_BASE = "https://api.llama.fi"
STABLECOINS_BASE = "https://stablecoins.llama.fi"
YIELDS_BASE = "https://yields.llama.fi"

TVL_HISTORY_URL = f"{LLAMA_BASE}/v2/historicalChainTvl"
CHAIN_TVL_URL = f"{LLAMA_BASE}/v2/historicalChainTvl/{{chain}}"
STABLECOINS_URL = f"{STABLECOINS_BASE}/stablecoins?includePrices=true"
YIELDS_URL = f"{YIELDS_BASE}/pools"

REQUEST_DELAY = 2.0       # seconds between requests
CACHE_DURATION = 1800     # 30 minutes — TVL updates ~hourly

# Chains to monitor individually
TRACKED_CHAINS = ["Ethereum", "Solana", "Arbitrum", "Base", "BNB"]

# Tickers affected by each chain's TVL movement
_CHAIN_TICKERS: Dict[str, List[str]] = {
    "Ethereum": ["ETH-USD", "ETHX"],
    "Solana": ["SOL-USD"],
    "Arbitrum": ["ETH-USD", "ARB-USD"],
    "Base": ["ETH-USD", "COIN"],
    "BNB": ["BNB-USD"],
    "Total": ["BTC-USD", "ETH-USD", "COIN", "MSTR"],
}

# Stablecoins that matter (broad crypto health indicators)
_KEY_STABLECOINS = {"USDT", "USDC", "DAI", "BUSD", "FRAX", "TUSD", "USDP"}

# TVL thresholds for signal generation
TVL_SURGE_THRESHOLD = 0.10    # 10% gain = significant inflow
TVL_FLIGHT_THRESHOLD = -0.10  # 10% drop = significant outflow
STABLECOIN_CHANGE_THRESHOLD = 0.05  # 5% stablecoin supply change = signal


@dataclass
class DefiSignal:
    """A DeFi capital flow signal from DefiLlama."""

    metric: str           # "tvl_change", "stablecoin_flow", "yield_spike"
    chain: str            # "Ethereum", "Solana", "Total"
    value: float          # Current value (USD or supply)
    change_24h: float     # 24h change as fraction (0.10 = 10%)
    change_7d: float      # 7d change as fraction
    direction: str        # "bullish" or "bearish"
    strength: float       # 0.0-1.0 signal strength
    confidence: float     # 0.0-1.0
    affected_tickers: List[str] = field(default_factory=list)
    signal_source: str = "defillama"
    domain: str = "defi"
    details: str = ""     # Human-readable context

    def to_plain_language(self) -> str:
        chg_24h_pct = self.change_24h * 100
        tickers = ", ".join(self.affected_tickers[:4]) if self.affected_tickers else "crypto"
        return (
            f"DefiLlama [{self.chain}] {self.metric}: "
            f"${self.value:,.0f} | "
            f"24h: {chg_24h_pct:+.1f}% | "
            f"{self.direction.upper()} for {tickers}"
        )


class DefiLlamaClient:
    """
    Client for DefiLlama DeFi capital flow data.

    Tracks TVL changes per chain and stablecoin supply flows.
    No auth required. Free API.

    Rate limited to 1 req/2s. 30-minute cache.
    """

    def __init__(self, provider=None, raw_store=None):
        self._provider = provider
        self._raw_store = raw_store
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MIDGE Trading Research",
            "Accept": "application/json",
        })
        self._last_request_time: float = 0.0
        self._tvl_cache: Optional[List[DefiSignal]] = None
        self._tvl_cache_time: float = 0.0
        self._stable_cache: Optional[List[DefiSignal]] = None
        self._stable_cache_time: float = 0.0
        self._call_count = 0
        self._error_count = 0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _fetch_json(self, url: str) -> Optional[object]:
        """Fetch JSON from DefiLlama APIs."""
        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            resp = market_request(
                self._provider, url,
                source_name="defillama", timeout_ms=20000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return resp.payload
            return None

        try:
            r = self.session.get(url, timeout=20)
            self._call_count += 1
            if r.status_code == 200:
                return r.json()
            logger.warning("DefiLlama HTTP %d for %s", r.status_code, url)
            return None
        except Exception as exc:
            self._error_count += 1
            logger.warning("DefiLlama fetch error: %s", exc)
            return None

    def get_tvl_changes(self) -> List[DefiSignal]:
        """Get 24h and 7d TVL changes for total DeFi and per-chain.

        TVL drop >10% = capital flight (bearish).
        TVL surge >10% = capital inflow (bullish).

        Returns:
            List of DefiSignal for chains with significant TVL movement.
        """
        if self._tvl_cache is not None and time.time() - self._tvl_cache_time < CACHE_DURATION:
            return self._tvl_cache

        signals = []

        # Total TVL first
        self._rate_limit()
        total_data = self._fetch_json(TVL_HISTORY_URL)
        if total_data:
            signal = self._compute_tvl_signal(total_data, "Total")
            if signal is not None:
                signals.append(signal)

        # Per-chain TVL
        for chain in TRACKED_CHAINS:
            self._rate_limit()
            chain_data = self._fetch_json(CHAIN_TVL_URL.format(chain=chain))
            if chain_data:
                signal = self._compute_tvl_signal(chain_data, chain)
                if signal is not None:
                    signals.append(signal)

        # Store raw data
        if self._raw_store is not None and signals:
            try:
                self._raw_store.store_defillama_tvl([
                    {
                        "chain": s.chain,
                        "value": s.value,
                        "change_24h": s.change_24h,
                        "change_7d": s.change_7d,
                        "direction": s.direction,
                    }
                    for s in signals
                ])
            except Exception as exc:
                logger.debug("RawStore DefiLlama TVL write failed: %s", exc)

        self._tvl_cache = signals
        self._tvl_cache_time = time.time()

        if signals:
            logger.info(
                "DefiLlama TVL: %d chain signals generated",
                len(signals),
            )

        return signals

    def get_stablecoin_flows(self) -> List[DefiSignal]:
        """Track stablecoin supply changes as proxy for capital entering/exiting crypto.

        USDC/USDT supply growing = new money entering crypto (bullish).
        Supply shrinking = capital leaving (bearish).

        Returns:
            List of DefiSignal for stablecoins with significant supply changes.
        """
        if self._stable_cache is not None and time.time() - self._stable_cache_time < CACHE_DURATION:
            return self._stable_cache

        self._rate_limit()
        data = self._fetch_json(STABLECOINS_URL)
        if not data:
            self._stable_cache = []
            self._stable_cache_time = time.time()
            return []

        signals = []

        # API returns {"peggedAssets": [...]}
        assets = []
        if isinstance(data, dict):
            assets = data.get("peggedAssets", data.get("stablecoins", []))
        elif isinstance(data, list):
            assets = data

        for asset in assets:
            signal = self._parse_stablecoin(asset)
            if signal is not None:
                signals.append(signal)

        # Also compute aggregate stablecoin signal (total supply change)
        total_signal = self._compute_aggregate_stablecoin(signals)
        if total_signal is not None:
            signals.insert(0, total_signal)

        # Store raw data
        if self._raw_store is not None and signals:
            try:
                self._raw_store.store_defillama_stablecoins([
                    {
                        "chain": s.chain,
                        "value": s.value,
                        "change_24h": s.change_24h,
                        "direction": s.direction,
                    }
                    for s in signals[:20]
                ])
            except Exception as exc:
                logger.debug("RawStore DefiLlama stablecoins write failed: %s", exc)

        self._stable_cache = signals
        self._stable_cache_time = time.time()

        if signals:
            logger.info(
                "DefiLlama stablecoins: %d supply signals",
                len(signals),
            )

        return signals

    def _compute_tvl_signal(
        self, history: object, chain: str
    ) -> Optional[DefiSignal]:
        """Compute TVL change signal from historical data array.

        DefiLlama returns: [{"date": timestamp, "tvl": float}, ...]
        """
        if not isinstance(history, list) or len(history) < 2:
            return None

        # Sort by date ascending (oldest first)
        try:
            sorted_data = sorted(history, key=lambda x: x.get("date", 0))
        except (TypeError, AttributeError):
            return None

        if not sorted_data:
            return None

        current = sorted_data[-1]
        current_tvl = float(current.get("tvl", 0.0))

        if current_tvl <= 0:
            return None

        # 24h ago — find entry closest to 24 datapoints back
        # DefiLlama daily data: ~1 point/day; hourly: ~24 points/day
        idx_24h = max(0, len(sorted_data) - 2)  # yesterday
        idx_7d = max(0, len(sorted_data) - 8)   # ~7 days back

        prev_24h = float(sorted_data[idx_24h].get("tvl", current_tvl))
        prev_7d = float(sorted_data[idx_7d].get("tvl", current_tvl))

        change_24h = (current_tvl - prev_24h) / prev_24h if prev_24h > 0 else 0.0
        change_7d = (current_tvl - prev_7d) / prev_7d if prev_7d > 0 else 0.0

        # Only emit signal if change is significant
        if abs(change_24h) < 0.03 and abs(change_7d) < 0.05:
            return None

        direction = "bullish" if change_24h >= 0 else "bearish"

        # Strength: scales with magnitude of change
        abs_change = abs(change_24h)
        strength = min(1.0, abs_change / TVL_SURGE_THRESHOLD)

        # Confidence: higher for larger chains (Total > Ethereum > others)
        chain_weight = {"Total": 0.8, "Ethereum": 0.7, "Solana": 0.6}.get(chain, 0.5)
        confidence = round(chain_weight * min(1.0, abs_change / 0.15), 3)

        tickers = _CHAIN_TICKERS.get(chain, ["BTC-USD", "ETH-USD"])
        details = (
            f"TVL ${current_tvl/1e9:.1f}B | "
            f"24h: {change_24h*100:+.1f}% | "
            f"7d: {change_7d*100:+.1f}%"
        )

        return DefiSignal(
            metric="tvl_change",
            chain=chain,
            value=round(current_tvl, 0),
            change_24h=round(change_24h, 4),
            change_7d=round(change_7d, 4),
            direction=direction,
            strength=round(strength, 3),
            confidence=confidence,
            affected_tickers=tickers[:],
            details=details,
        )

    def _parse_stablecoin(self, asset: dict) -> Optional[DefiSignal]:
        """Parse a single stablecoin asset from DefiLlama stablecoins API."""
        symbol = (asset.get("symbol") or asset.get("name") or "").upper()

        # Only track key stablecoins
        if symbol not in _KEY_STABLECOINS:
            return None

        # Current circulating supply (USD)
        circulating = asset.get("circulating", {})
        if isinstance(circulating, dict):
            current_supply = float(circulating.get("peggedUSD", 0.0))
        else:
            current_supply = float(circulating or 0.0)

        if current_supply <= 0:
            return None

        # Change fields — DefiLlama provides 1d/7d changes
        change_1d = float(asset.get("change_1d") or 0.0) / 100.0
        change_7d = float(asset.get("change_7d") or 0.0) / 100.0

        # Filter — only signal if supply moved significantly
        if abs(change_1d) < STABLECOIN_CHANGE_THRESHOLD and abs(change_7d) < 0.08:
            return None

        direction = "bullish" if change_1d >= 0 else "bearish"
        strength = min(1.0, abs(change_1d) / 0.10)
        confidence = round(0.5 * min(1.0, current_supply / 10_000_000_000), 3)  # $10B = full weight

        details = (
            f"{symbol} supply ${current_supply/1e9:.1f}B | "
            f"24h: {change_1d*100:+.1f}%"
        )

        return DefiSignal(
            metric="stablecoin_flow",
            chain=symbol,
            value=round(current_supply, 0),
            change_24h=round(change_1d, 4),
            change_7d=round(change_7d, 4),
            direction=direction,
            strength=round(strength, 3),
            confidence=confidence,
            affected_tickers=["BTC-USD", "ETH-USD", "COIN"],
            details=details,
        )

    def _compute_aggregate_stablecoin(
        self, individual_signals: List[DefiSignal]
    ) -> Optional[DefiSignal]:
        """Compute an aggregate stablecoin signal from individual stablecoin signals.

        Returns None if fewer than 2 signals available.
        """
        if len(individual_signals) < 2:
            return None

        total_supply = sum(s.value for s in individual_signals)
        if total_supply <= 0:
            return None

        # Supply-weighted 24h change
        weighted_change = sum(
            s.change_24h * s.value for s in individual_signals
        ) / total_supply

        if abs(weighted_change) < STABLECOIN_CHANGE_THRESHOLD:
            return None

        direction = "bullish" if weighted_change >= 0 else "bearish"
        strength = min(1.0, abs(weighted_change) / 0.10)

        details = (
            f"Aggregate stablecoin supply ${total_supply/1e9:.1f}B | "
            f"24h weighted: {weighted_change*100:+.1f}%"
        )

        return DefiSignal(
            metric="stablecoin_flow",
            chain="Total_Stablecoins",
            value=round(total_supply, 0),
            change_24h=round(weighted_change, 4),
            change_7d=0.0,
            direction=direction,
            strength=round(strength, 3),
            confidence=0.75,  # Aggregate is more reliable than individual
            affected_tickers=["BTC-USD", "ETH-USD", "SOL-USD", "COIN"],
            details=details,
        )

    def get_statistics(self) -> dict:
        """Return client statistics."""
        return {
            "calls_made": self._call_count,
            "errors": self._error_count,
            "tvl_signals_cached": len(self._tvl_cache) if self._tvl_cache else 0,
            "stablecoin_signals_cached": len(self._stable_cache) if self._stable_cache else 0,
        }


def get_tvl_changes() -> List[DefiSignal]:
    """Convenience function for one-shot TVL change fetch."""
    return DefiLlamaClient().get_tvl_changes()


def get_stablecoin_flows() -> List[DefiSignal]:
    """Convenience function for one-shot stablecoin flow fetch."""
    return DefiLlamaClient().get_stablecoin_flows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Testing DefiLlama client...")

    client = DefiLlamaClient()

    print("\nTVL Changes:")
    tvl = client.get_tvl_changes()
    for s in tvl[:5]:
        print(f"  {s.to_plain_language()}")

    print("\nStablecoin Flows:")
    flows = client.get_stablecoin_flows()
    for s in flows[:5]:
        print(f"  {s.to_plain_language()}")

    print(f"\nStats: {client.get_statistics()}")
    print("\nDone.")
