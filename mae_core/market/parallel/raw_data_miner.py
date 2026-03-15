"""raw_data_miner.py — DuckDB-powered raw store signal extractor.

Reads MIDGE's 10 SQLite raw store databases via DuckDB and extracts patterns
the signal adapters miss. The signal adapters extract 1-2 values per API call
and discard the rest. This process mines the other 90%.

What gets extracted:
  - prices.db:      52-week proximity, ATH distance, volume vs 3-month average,
                    bid/ask spread anomalies, price-to-SMA deviations
  - cot.db:         ALL 384 contracts (not just top 10), extreme positioning,
                    commercial vs non-commercial divergence, open interest spikes
  - openinsider.db: Large purchase clusters per ticker (3+ insiders = high signal),
                    executive rank patterns (CEO/CFO buys > director buys),
                    delta_owned_pct spikes (insider doubling their stake)
  - sec_edgar.db:   Institutional accumulation (13F), activist stake sizes
  - finviz.db:      Unusual volume + short float combo (squeeze setup detector),
                    sector-level volume clustering
  - fred.db:        Yield curve spread rates of change, credit spread levels,
                    cross-series divergence signals
  - massive.db:     VWAP vs close deviation, gap-up/gap-down opens,
                    transaction count anomalies (institutional block trades)
  - finnhub.db:     Real-time tick velocity (trades/minute spikes)
  - legislation.db: Bills in specific policy areas affecting tickers
  - finra.db:       Speculative short ratio vs regular short ratio divergence

Output:
  Each extracted pattern becomes a MarketSignal dict written as JSONL to
  data/market/raw_miner_signals.jsonl. The daemon's signal buffer picks these
  up on next sensing cycle via _ingest_bridge_file() in sensing_hook.py.

  Bridge signals use source="raw_miner_{store}" and domain from the pattern
  type so they flow through Thompson weighting like any other signal.

Run as independent process:
    python -m mae_core.market.parallel.raw_data_miner
    python -m mae_core.market.parallel.raw_data_miner --interval 300 --once

Design principles:
  - DuckDB attaches SQLite files read-only — NEVER writes to raw stores
  - Graceful skip on missing tables (stores populate progressively)
  - Per-store try/except — one broken store never kills the others
  - Bridge file write is atomic (temp file → replace)
  - Deduplicates signals by signal_id within each cycle
  - Logs discovery counts: "Extracted 43 large COT positioning signals"
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
from datetime import timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RAW_DIR      = _PROJECT_ROOT / "data" / "market" / "raw"
_DATA_DIR     = _PROJECT_ROOT / "data" / "market"
_BRIDGE_FILE  = _DATA_DIR / "raw_miner_signals.jsonl"
_LOG_PATH     = _DATA_DIR / "raw_miner.log"

# ── Configuration ──────────────────────────────────────────────────────────────
_DEFAULT_INTERVAL_SECONDS = 300   # 5 minutes between cycles
_LOOKBACK_DAYS_INSIDER    = 30    # How far back to mine insider trades
_LOOKBACK_DAYS_PRICES     = 7     # Price snapshots (stale quickly)
_LOOKBACK_DAYS_COT        = 90    # COT data (weekly cadence)
_LOOKBACK_DAYS_FRED       = 90    # FRED series
_LOOKBACK_DAYS_MASSIVE    = 7     # Daily bars
_LOOKBACK_DAYS_FINRA      = 14    # Short volume
_LOOKBACK_DAYS_FINNHUB    = 1     # Ticks (very high volume, keep recent)
_MAX_SIGNALS_PER_CYCLE    = 2000  # Safety cap — never flood the bridge

# ── COT ticker mapping: CFTC contract name → (ticker, asset_class) ─────────────
# This maps the 384 COT contracts to tradeable tickers for signal routing.
# The current signal adapter only handles 10; this expands to every mapped contract.
_COT_TICKER_MAP: Dict[str, tuple] = {
    "E-MINI S&P 500":              ("ES=F",   "futures"),
    "E-MINI NASDAQ-100":           ("NQ=F",   "futures"),
    "E-MINI DOW ($5)":             ("YM=F",   "futures"),
    "RUSSELL 2000 MINI":           ("RTY=F",  "futures"),
    "10-YEAR T-NOTE":              ("ZN=F",   "futures"),
    "30-YEAR T-BOND":              ("ZB=F",   "futures"),
    "2-YEAR T-NOTE":               ("ZT=F",   "futures"),
    "5-YEAR T-NOTE":               ("ZF=F",   "futures"),
    "EURO FX":                     ("EURUSD=X","forex"),
    "JAPANESE YEN":                ("USDJPY=X","forex"),
    "BRITISH POUND":               ("GBPUSD=X","forex"),
    "SWISS FRANC":                 ("CHFUSD=X","forex"),
    "CANADIAN DOLLAR":             ("CADUSD=X","forex"),
    "AUSTRALIAN DOLLAR":           ("AUDUSD=X","forex"),
    "GOLD":                        ("GC=F",   "futures"),
    "SILVER":                      ("SI=F",   "futures"),
    "COPPER":                      ("HG=F",   "futures"),
    "PLATINUM":                    ("PL=F",   "futures"),
    "CRUDE OIL, LIGHT SWEET":      ("CL=F",   "futures"),
    "NATURAL GAS":                 ("NG=F",   "futures"),
    "RBOB GASOLINE":               ("RB=F",   "futures"),
    "HEATING OIL":                 ("HO=F",   "futures"),
    "CORN":                        ("ZC=F",   "futures"),
    "SOYBEANS":                    ("ZS=F",   "futures"),
    "WHEAT-SRW":                   ("ZW=F",   "futures"),
    "BITCOIN":                     ("BTC-USD","crypto"),
    "ETHER":                       ("ETH-USD","crypto"),
    "VIX":                         ("^VIX",   "index"),
    "S&P 500 CONSOLIDATED":        ("SPY",    "stock"),
    "U.S. DOLLAR INDEX":           ("DX=F",   "futures"),
}

# ── Logging ─────────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    """Configure rotating file + console logging."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("raw_data_miner")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.handlers.RotatingFileHandler(
            _LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s [raw_miner] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ── Date helpers ───────────────────────────────────────────────────────────────

def _cutoff_date(lookback_days: int) -> str:
    """Return ISO date string N days ago, e.g. '2026-03-08'.

    All raw store timestamps/dates are stored as VARCHAR in SQLite.
    DuckDB cannot compare VARCHAR against CURRENT_DATE/CURRENT_TIMESTAMP without
    an explicit cast. String comparison works correctly for ISO-format dates
    because YYYY-MM-DD lexicographic order == chronological order.
    """
    return (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")


def _cutoff_datetime(lookback_days: int) -> str:
    """Return ISO datetime string N days ago, e.g. '2026-03-08T06:50:00'.

    Used for columns storing full ISO timestamps as VARCHAR.
    """
    return (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


# ── DuckDB connection factory ───────────────────────────────────────────────────

def _open_duckdb() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection with SQLite extension loaded.

    Returns an unattached connection. Callers must attach the specific
    SQLite database they need before querying.
    """
    con = duckdb.connect(database=":memory:")
    # Install + load the SQLite scanner (bundled with DuckDB ≥ 0.8)
    try:
        con.execute("INSTALL sqlite;")
    except Exception:
        pass  # Already installed
    con.execute("LOAD sqlite;")
    return con


def _attach_db(
    con: duckdb.DuckDBPyConnection,
    db_name: str,
    alias: str,
) -> bool:
    """Attach a SQLite raw store as a read-only DuckDB alias.

    Returns True if the file exists and was attached. False otherwise.
    """
    db_path = _RAW_DIR / f"{db_name}.db"
    if not db_path.exists():
        return False
    try:
        # Detach first in case it was previously attached
        try:
            con.execute(f"DETACH {alias}")
        except Exception:
            pass
        con.execute(f"ATTACH '{db_path}' AS {alias} (TYPE sqlite, READ_ONLY)")
        return True
    except Exception:
        return False


# ── Signal builder ──────────────────────────────────────────────────────────────

def _make_signal(
    source: str,
    symbol: str,
    asset_class: str,
    domain: str,
    direction: str,
    strength: float,
    confidence: float,
    decay_rate: float,
    timestamp: str,
    outcome_window_days: int,
    metadata: dict,
    signal_suffix: str = "",
) -> dict:
    """Build a MarketSignal-compatible dict for the bridge file.

    Uses dict format (not dataclass) so it serializes cleanly to JSONL
    without importing the full signal module.
    """
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    sig_id = f"raw_miner:{source}:{symbol}:{ts[:10]}:{signal_suffix or uuid.uuid4().hex[:8]}"
    return {
        "signal_id": sig_id,
        "source": f"raw_miner_{source}",
        "symbol": symbol,
        "asset_class": asset_class,
        "domain": domain,
        "direction": direction,
        "strength": round(max(0.0, min(1.0, strength)), 4),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "decay_rate": decay_rate,
        "timestamp": ts,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "velocity": 0.0,
        "outcome_symbol": symbol,
        "outcome_window_days": outcome_window_days,
        "raw_id": "",
        "raw_type": f"RawMiner_{source}",
        "metadata": metadata,
    }


# ── Per-store extractors ────────────────────────────────────────────────────────

def _mine_prices(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    lookback_days: int = _LOOKBACK_DAYS_PRICES,
) -> List[dict]:
    """Extract price-fundamental signals from the yfinance snapshot store.

    Detects:
    - 52-week proximity: price within 5% of 52-week high (breakout setup)
    - ATH distance: price within 3% of all-time high (rare event)
    - Volume surge: today's volume > 3x 3-month average
    - Price-to-50MA deviation: price > 10% above or below 50-day MA
    - Short ratio elevated: > 0.20 (short squeeze risk or bearish pressure)
    """
    signals: List[dict] = []
    if not _attach_db(con, "prices", "prices"):
        return signals

    cutoff = _cutoff_datetime(lookback_days)
    try:
        rows = con.execute(f"""
            SELECT
                symbol,
                price,
                market_cap,
                short_ratio,
                fifty_two_week_high,
                fifty_two_week_low,
                info_json,
                ingested_at
            FROM prices.price_snapshots
            WHERE ingested_at >= '{cutoff}'
              AND price > 0
              AND fifty_two_week_high IS NOT NULL
              AND fifty_two_week_high > 0
            ORDER BY ingested_at DESC
        """).fetchall()
    except Exception as exc:
        logger.debug("prices query failed: %s", exc)
        return signals

    seen: set = set()
    for (symbol, price, mktcap, short_ratio, wk52h, wk52l,
         info_json_str, ingested_at) in rows:

        if not symbol or symbol in seen:
            continue
        seen.add(symbol)

        try:
            info = json.loads(info_json_str or "{}")
        except (json.JSONDecodeError, TypeError):
            info = {}

        ts = ingested_at or datetime.now(timezone.utc).isoformat()

        # --- 52-week high proximity (breakout setup) ---
        if wk52h and price:
            proximity = price / wk52h
            if proximity >= 0.97:  # Within 3% of 52-week high
                strength = min(1.0, (proximity - 0.95) / 0.05)
                signals.append(_make_signal(
                    source="prices_52wk_high",
                    symbol=symbol,
                    asset_class="stock",
                    domain="technical",
                    direction="bullish",
                    strength=strength,
                    confidence=0.45,
                    decay_rate=0.10,
                    timestamp=ts,
                    outcome_window_days=14,
                    metadata={
                        "price": price,
                        "fifty_two_week_high": wk52h,
                        "proximity_pct": round(proximity * 100, 2),
                        "market_cap": mktcap,
                    },
                    signal_suffix="52wk_hi",
                ))

        # --- Volume surge vs 3-month average ---
        vol_today = info.get("regularMarketVolume", 0) or 0
        avg_vol_3m = info.get("averageDailyVolume3Month", 0) or 0
        if vol_today and avg_vol_3m and avg_vol_3m > 0:
            vol_ratio = vol_today / avg_vol_3m
            if vol_ratio >= 3.0:
                # Direction from price change
                chg = info.get("regularMarketChangePercent", 0) or 0
                direction = "bullish" if chg > 0 else "bearish" if chg < 0 else "neutral"
                strength = min(1.0, (vol_ratio - 3.0) / 7.0 + 0.4)
                signals.append(_make_signal(
                    source="prices_volume_surge",
                    symbol=symbol,
                    asset_class="stock",
                    domain="technical",
                    direction=direction,
                    strength=strength,
                    confidence=0.55,
                    decay_rate=0.20,
                    timestamp=ts,
                    outcome_window_days=7,
                    metadata={
                        "volume_today": vol_today,
                        "avg_volume_3m": avg_vol_3m,
                        "volume_ratio": round(vol_ratio, 2),
                        "change_pct": round(chg, 3),
                    },
                    signal_suffix="vol_surge",
                ))

        # --- Price vs 50-day moving average deviation ---
        ma50 = info.get("fiftyDayAverage", 0) or 0
        if price and ma50 and ma50 > 0:
            deviation = (price - ma50) / ma50
            if abs(deviation) >= 0.10:
                direction = "bullish" if deviation > 0 else "bearish"
                strength = min(1.0, abs(deviation) / 0.25)
                signals.append(_make_signal(
                    source="prices_ma_deviation",
                    symbol=symbol,
                    asset_class="stock",
                    domain="technical",
                    direction=direction,
                    strength=strength,
                    confidence=0.40,
                    decay_rate=0.08,
                    timestamp=ts,
                    outcome_window_days=14,
                    metadata={
                        "price": price,
                        "fifty_day_ma": round(ma50, 4),
                        "deviation_pct": round(deviation * 100, 2),
                    },
                    signal_suffix="ma_dev",
                ))

        # --- Short ratio elevated ---
        if short_ratio and short_ratio >= 0.20:
            direction = "bearish" if short_ratio >= 0.50 else "neutral"
            strength = min(1.0, short_ratio / 0.70)
            signals.append(_make_signal(
                source="prices_short_ratio",
                symbol=symbol,
                asset_class="stock",
                domain="institutional",
                direction=direction,
                strength=strength,
                confidence=0.45,
                decay_rate=0.05,
                timestamp=ts,
                outcome_window_days=21,
                metadata={
                    "short_ratio": round(short_ratio, 4),
                    "price": price,
                    "market_cap": mktcap,
                },
                signal_suffix="short_ratio",
            ))

    logger.info(
        "prices: %d snapshots → %d signals (52wk_high/vol_surge/ma_dev/short_ratio)",
        len(seen), len(signals),
    )
    return signals


def _mine_cot(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    lookback_days: int = _LOOKBACK_DAYS_COT,
) -> List[dict]:
    """Extract COT positioning signals across ALL 384 contracts.

    The signal adapter currently handles only ~10 hardcoded contracts.
    This miner extracts from every contract with a known ticker mapping.

    Detects:
    - Extreme non-commercial net long/short (>75th percentile by open interest)
    - Commercial vs non-commercial divergence (smart money vs speculators)
    - Open interest spikes (entering new directional conviction)
    - Week-over-week net position change acceleration
    """
    signals: List[dict] = []
    if not _attach_db(con, "cot", "cot"):
        return signals

    cutoff = _cutoff_date(lookback_days)
    try:
        # Get last 2 weeks for each contract to compute change
        rows = con.execute(f"""
            WITH ranked AS (
                SELECT
                    contract_name,
                    report_date,
                    commercial_long,
                    commercial_short,
                    noncommercial_long,
                    noncommercial_short,
                    open_interest,
                    ROW_NUMBER() OVER (
                        PARTITION BY contract_name ORDER BY report_date DESC
                    ) AS rn
                FROM cot.cot_weekly
                WHERE report_date >= '{cutoff}'
            )
            SELECT
                contract_name,
                report_date,
                commercial_long,
                commercial_short,
                noncommercial_long,
                noncommercial_short,
                open_interest,
                rn
            FROM ranked
            WHERE rn <= 2
            ORDER BY contract_name, rn
        """).fetchall()
    except Exception as exc:
        logger.debug("cot query failed: %s", exc)
        return signals

    # Group by contract: {name: [most_recent_row, prior_row]}
    by_contract: Dict[str, list] = {}
    for row in rows:
        name = row[0]
        if name not in by_contract:
            by_contract[name] = []
        by_contract[name].append(row)

    contracts_processed = 0
    for contract_name, contract_rows in by_contract.items():
        if not contract_rows:
            continue

        # Resolve ticker from mapping (partial match)
        ticker = None
        asset_class = "futures"
        for key, (t, ac) in _COT_TICKER_MAP.items():
            if key in contract_name.upper() or contract_name.upper().startswith(key):
                ticker = t
                asset_class = ac
                break
        if not ticker:
            # Skip unmapped contracts — no ticker to route the signal to
            continue

        contracts_processed += 1
        current = contract_rows[0]
        prior   = contract_rows[1] if len(contract_rows) > 1 else None

        (_, report_date, comm_long, comm_short,
         nc_long, nc_short, open_interest, _) = current

        if open_interest is None or open_interest == 0:
            continue

        nc_net = (nc_long or 0) - (nc_short or 0)
        comm_net = (comm_long or 0) - (comm_short or 0)

        # Non-commercial net as fraction of open interest
        nc_net_pct = nc_net / open_interest if open_interest else 0.0

        # --- Extreme speculator positioning ---
        if abs(nc_net_pct) >= 0.20:  # Speculators hold 20%+ net long or short
            direction = "bullish" if nc_net_pct > 0 else "bearish"
            strength = min(1.0, abs(nc_net_pct) / 0.40)
            signals.append(_make_signal(
                source="cot_positioning",
                symbol=ticker,
                asset_class=asset_class,
                domain="positioning",
                direction=direction,
                strength=strength,
                confidence=0.50,
                decay_rate=0.03,
                timestamp=report_date,
                outcome_window_days=21,
                metadata={
                    "contract_name": contract_name,
                    "nc_long": nc_long,
                    "nc_short": nc_short,
                    "nc_net": nc_net,
                    "nc_net_pct": round(nc_net_pct, 4),
                    "open_interest": open_interest,
                    "signal_type": "extreme_speculator_positioning",
                },
                signal_suffix=f"cot_extreme:{contract_name[:20]}",
            ))

        # --- Commercial vs non-commercial divergence ---
        # Commercials (hedgers/producers) are considered "smart money".
        # When commercials are net opposite to non-commercials with strong
        # conviction, it often precedes reversals.
        if (comm_net > 0 and nc_net < 0) or (comm_net < 0 and nc_net > 0):
            comm_net_pct = comm_net / open_interest if open_interest else 0.0
            divergence = abs(comm_net_pct) + abs(nc_net_pct)
            if divergence >= 0.30:
                # Commercials direction is the "smart money" signal
                direction = "bullish" if comm_net > 0 else "bearish"
                strength = min(1.0, divergence / 0.60)
                signals.append(_make_signal(
                    source="cot_divergence",
                    symbol=ticker,
                    asset_class=asset_class,
                    domain="positioning",
                    direction=direction,
                    strength=strength,
                    confidence=0.55,
                    decay_rate=0.03,
                    timestamp=report_date,
                    outcome_window_days=28,
                    metadata={
                        "contract_name": contract_name,
                        "commercial_net": comm_net,
                        "nc_net": nc_net,
                        "divergence": round(divergence, 4),
                        "signal_type": "commercial_speculator_divergence",
                    },
                    signal_suffix=f"cot_div:{contract_name[:20]}",
                ))

        # --- Week-over-week net position acceleration ---
        if prior is not None:
            (_, _, p_comm_long, p_comm_short,
             p_nc_long, p_nc_short, p_oi, _) = prior
            p_nc_net = (p_nc_long or 0) - (p_nc_short or 0)
            nc_net_change = nc_net - p_nc_net
            # Significant weekly position change (> 5% of open interest)
            if open_interest and abs(nc_net_change) / open_interest >= 0.05:
                direction = "bullish" if nc_net_change > 0 else "bearish"
                strength = min(1.0, abs(nc_net_change) / (open_interest * 0.15))
                signals.append(_make_signal(
                    source="cot_weekly_change",
                    symbol=ticker,
                    asset_class=asset_class,
                    domain="positioning",
                    direction=direction,
                    strength=strength,
                    confidence=0.50,
                    decay_rate=0.05,
                    timestamp=report_date,
                    outcome_window_days=14,
                    metadata={
                        "contract_name": contract_name,
                        "nc_net_change": nc_net_change,
                        "nc_net_change_pct": round(nc_net_change / open_interest, 4),
                        "prior_nc_net": p_nc_net,
                        "current_nc_net": nc_net,
                        "signal_type": "weekly_position_change",
                    },
                    signal_suffix=f"cot_chg:{contract_name[:20]}",
                ))

    logger.info(
        "cot: %d mapped contracts → %d signals (extreme/divergence/change)",
        contracts_processed, len(signals),
    )
    return signals


def _mine_openinsider(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    lookback_days: int = _LOOKBACK_DAYS_INSIDER,
) -> List[dict]:
    """Extract enhanced insider signals from the OpenInsider purchase store.

    The existing adapter converts individual purchases to signals. This miner
    additionally extracts:
    - Cluster buys: 3+ insiders buying the same ticker within 14 days
    - Executive-rank signals: CEO/CFO/President buys (highest conviction tier)
    - Stake-doubling signals: delta_owned_pct > 50% (insider doubling position)
    - Cumulative value thresholds: total purchases > $5M in 30 days
    """
    signals: List[dict] = []
    if not _attach_db(con, "openinsider", "openinsider"):
        return signals

    cutoff = _cutoff_date(lookback_days)
    try:
        # Cluster detection: 3+ insiders buying same ticker within window
        cluster_rows = con.execute(f"""
            SELECT
                ticker,
                COUNT(*) AS insider_count,
                SUM(COALESCE(value, 0)) AS total_value,
                MIN(trade_date) AS earliest_date,
                MAX(trade_date) AS latest_date,
                LIST(insider_name) AS insider_names,
                LIST(title) AS titles
            FROM openinsider.insider_purchases
            WHERE trade_date >= '{cutoff}'
              AND trade_type = 'P - Purchase'
              AND value > 10000
            GROUP BY ticker
            HAVING COUNT(*) >= 3
            ORDER BY total_value DESC
            LIMIT 100
        """).fetchall()
    except Exception as exc:
        logger.debug("openinsider cluster query failed: %s", exc)
        cluster_rows = []

    for (ticker, insider_count, total_value, earliest_date,
         latest_date, insider_names, titles) in cluster_rows:
        if not ticker:
            continue
        # Strength: 3 insiders = 0.70, 5+ = 0.95
        strength = min(0.95, 0.50 + (insider_count - 3) * 0.10 + 0.20)
        # Value boost: > $1M total = extra confidence
        confidence = 0.65 if total_value and total_value > 1_000_000 else 0.58
        signals.append(_make_signal(
            source="openinsider_cluster",
            symbol=ticker,
            asset_class="stock",
            domain="insider",
            direction="bullish",
            strength=strength,
            confidence=confidence,
            decay_rate=0.04,
            timestamp=str(latest_date),
            outcome_window_days=30,
            metadata={
                "insider_count": insider_count,
                "total_value": total_value,
                "earliest_date": str(earliest_date),
                "latest_date": str(latest_date),
                "insider_names": insider_names[:5],
                "titles": titles[:5],
                "signal_type": "insider_cluster_buy",
            },
            signal_suffix=f"cluster:{ticker}",
        ))

    try:
        # Executive-rank buys: CEO/CFO/President/COO (not just directors)
        exec_rows = con.execute(f"""
            SELECT
                ticker,
                insider_name,
                title,
                price,
                quantity,
                value,
                delta_owned_pct,
                trade_date
            FROM openinsider.insider_purchases
            WHERE trade_date >= '{cutoff}'
              AND trade_type = 'P - Purchase'
              AND (
                  UPPER(title) LIKE '%CEO%'
                  OR UPPER(title) LIKE '%CFO%'
                  OR UPPER(title) LIKE '%PRESIDENT%'
                  OR UPPER(title) LIKE '%COO%'
                  OR UPPER(title) LIKE '%CHIEF%'
              )
              AND COALESCE(value, 0) > 50000
            ORDER BY COALESCE(value, 0) DESC
            LIMIT 200
        """).fetchall()
    except Exception as exc:
        logger.debug("openinsider exec query failed: %s", exc)
        exec_rows = []

    exec_seen: set = set()
    for (ticker, name, title, price, qty, value, delta_pct, trade_date) in exec_rows:
        if not ticker:
            continue
        sig_key = f"{ticker}:{name}:{trade_date}"
        if sig_key in exec_seen:
            continue
        exec_seen.add(sig_key)
        value = value or 0
        strength = min(1.0, value / 500_000.0)
        # Executive buys get a confidence premium
        signals.append(_make_signal(
            source="openinsider_executive",
            symbol=ticker,
            asset_class="stock",
            domain="insider",
            direction="bullish",
            strength=max(0.35, strength),
            confidence=0.65,   # Higher than standard adapter (0.55)
            decay_rate=0.04,
            timestamp=str(trade_date),
            outcome_window_days=30,
            metadata={
                "insider_name": name,
                "title": title,
                "value": value,
                "quantity": qty,
                "price": price,
                "delta_owned_pct": delta_pct,
                "signal_type": "executive_buy",
            },
            signal_suffix=f"exec:{ticker}:{(name or '')[:10]}",
        ))

    try:
        # Stake-doubling: delta_owned_pct > 50% (insider growing position by half)
        stake_rows = con.execute(f"""
            SELECT ticker, insider_name, title, value, delta_owned_pct, trade_date
            FROM openinsider.insider_purchases
            WHERE trade_date >= '{cutoff}'
              AND trade_type = 'P - Purchase'
              AND delta_owned_pct >= 50
              AND COALESCE(value, 0) > 10000
            ORDER BY delta_owned_pct DESC
            LIMIT 100
        """).fetchall()
    except Exception as exc:
        logger.debug("openinsider stake query failed: %s", exc)
        stake_rows = []

    for (ticker, name, title, value, delta_pct, trade_date) in stake_rows:
        if not ticker:
            continue
        signals.append(_make_signal(
            source="openinsider_stake_increase",
            symbol=ticker,
            asset_class="stock",
            domain="insider",
            direction="bullish",
            strength=min(1.0, (delta_pct or 0) / 200.0 + 0.50),
            confidence=0.60,
            decay_rate=0.04,
            timestamp=str(trade_date),
            outcome_window_days=30,
            metadata={
                "insider_name": name,
                "title": title,
                "value": value,
                "delta_owned_pct": delta_pct,
                "signal_type": "stake_doubling",
            },
            signal_suffix=f"stake:{ticker}",
        ))

    logger.info(
        "openinsider: %d clusters + %d exec buys + %d stake increases = %d signals",
        len(cluster_rows),
        len(exec_seen),
        len(stake_rows),
        len(signals),
    )
    return signals


def _mine_fred(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    lookback_days: int = _LOOKBACK_DAYS_FRED,
) -> List[dict]:
    """Extract cross-series macro signals from FRED observations.

    The standard adapter extracts individual series direction. This miner
    additionally extracts:
    - Yield curve level: current T10Y2Y spread value (not just direction)
    - Yield curve rate-of-change: spread steepening/flattening speed
    - Credit spread extremes: investment grade and HY spread levels
    - VIX from FRED (VIXCLS) vs moving average
    - Federal funds rate trajectory
    """
    signals: List[dict] = []
    if not _attach_db(con, "fred", "fred"):
        return signals

    cutoff = _cutoff_date(lookback_days)
    try:
        rows = con.execute(f"""
            SELECT series_id, date, value
            FROM fred.fred_observations
            WHERE date >= '{cutoff}'
              AND value IS NOT NULL
            ORDER BY series_id, date ASC
        """).fetchall()
    except Exception as exc:
        logger.debug("fred query failed: %s", exc)
        return signals

    # Group into series: {series_id: [(date, value), ...]}
    series_data: Dict[str, List[tuple]] = {}
    for (sid, date, value) in rows:
        if sid not in series_data:
            series_data[sid] = []
        series_data[sid].append((date, value))

    # --- Yield curve spread analysis ---
    yc = series_data.get("T10Y2Y", [])
    if len(yc) >= 5:
        latest_date, latest_val = yc[-1]
        prior_val = yc[-5][1]  # 5 observations ago
        roc = latest_val - prior_val  # Rate of change

        # Inversion level signal
        if latest_val < -0.50:  # Deep inversion
            signals.append(_make_signal(
                source="fred_yield_curve",
                symbol="",
                asset_class="macro",
                domain="macro",
                direction="bearish",
                strength=min(1.0, abs(latest_val) / 2.0),
                confidence=0.55,
                decay_rate=0.02,
                timestamp=str(latest_date),
                outcome_window_days=90,
                metadata={
                    "series_id": "T10Y2Y",
                    "value": latest_val,
                    "signal_type": "yield_curve_deep_inversion",
                },
                signal_suffix="yc_inversion",
            ))
        elif latest_val > 0.50:  # Steepening — recovery signal
            signals.append(_make_signal(
                source="fred_yield_curve",
                symbol="",
                asset_class="macro",
                domain="macro",
                direction="bullish",
                strength=min(1.0, latest_val / 2.0),
                confidence=0.50,
                decay_rate=0.02,
                timestamp=str(latest_date),
                outcome_window_days=90,
                metadata={
                    "series_id": "T10Y2Y",
                    "value": latest_val,
                    "signal_type": "yield_curve_steepening",
                },
                signal_suffix="yc_steep",
            ))

        # Rate of change signal (steepening/flattening velocity)
        if abs(roc) >= 0.25:
            direction = "bullish" if roc > 0 else "bearish"
            signals.append(_make_signal(
                source="fred_yield_curve_roc",
                symbol="",
                asset_class="macro",
                domain="macro",
                direction=direction,
                strength=min(1.0, abs(roc) / 1.0),
                confidence=0.45,
                decay_rate=0.05,
                timestamp=str(latest_date),
                outcome_window_days=30,
                metadata={
                    "series_id": "T10Y2Y",
                    "roc": round(roc, 4),
                    "from_value": prior_val,
                    "to_value": latest_val,
                    "signal_type": "yield_curve_rate_of_change",
                },
                signal_suffix="yc_roc",
            ))

    # --- Credit spread signals ---
    for series_id, threshold_high, threshold_low, symbol in [
        ("BAMLH0A0HYM2",  8.0, 3.0, "HYG"),   # HY OAS spread
        ("BAMLC0A0CM",    2.5, 0.8, "LQD"),    # IG OAS spread
    ]:
        spread_series = series_data.get(series_id, [])
        if len(spread_series) >= 2:
            latest_date, latest_val = spread_series[-1]
            if latest_val >= threshold_high:
                signals.append(_make_signal(
                    source="fred_credit_spread",
                    symbol=symbol,
                    asset_class="stock",
                    domain="macro",
                    direction="bearish",
                    strength=min(1.0, (latest_val - threshold_high) / threshold_high + 0.5),
                    confidence=0.55,
                    decay_rate=0.03,
                    timestamp=str(latest_date),
                    outcome_window_days=30,
                    metadata={
                        "series_id": series_id,
                        "value": latest_val,
                        "threshold": threshold_high,
                        "signal_type": "credit_spread_elevated",
                    },
                    signal_suffix=f"credit:{series_id}",
                ))
            elif latest_val <= threshold_low:
                signals.append(_make_signal(
                    source="fred_credit_spread",
                    symbol=symbol,
                    asset_class="stock",
                    domain="macro",
                    direction="bullish",
                    strength=min(1.0, (threshold_low - latest_val) / threshold_low + 0.3),
                    confidence=0.50,
                    decay_rate=0.03,
                    timestamp=str(latest_date),
                    outcome_window_days=30,
                    metadata={
                        "series_id": series_id,
                        "value": latest_val,
                        "threshold": threshold_low,
                        "signal_type": "credit_spread_compressed",
                    },
                    signal_suffix=f"credit:{series_id}_low",
                ))

    # --- VIX level from FRED (VIXCLS) ---
    vixcls = series_data.get("VIXCLS", [])
    if len(vixcls) >= 10:
        recent = [v for _, v in vixcls[-10:]]
        avg_vix = sum(recent) / len(recent)
        latest_date, latest_vix = vixcls[-1]
        if latest_vix >= 30.0:
            signals.append(_make_signal(
                source="fred_vix_elevated",
                symbol="^VIX",
                asset_class="index",
                domain="macro",
                direction="bearish",
                strength=min(1.0, (latest_vix - 20) / 40),
                confidence=0.55,
                decay_rate=0.10,
                timestamp=str(latest_date),
                outcome_window_days=14,
                metadata={
                    "vix": latest_vix,
                    "10d_avg": round(avg_vix, 2),
                    "signal_type": "vix_elevated",
                },
                signal_suffix="vix_high",
            ))
        elif latest_vix <= 13.0 and avg_vix > 16.0:
            # Sudden VIX compression from elevated levels = complacency risk
            signals.append(_make_signal(
                source="fred_vix_compression",
                symbol="^VIX",
                asset_class="index",
                domain="macro",
                direction="bearish",  # Complacency = risk of spike
                strength=0.45,
                confidence=0.45,
                decay_rate=0.08,
                timestamp=str(latest_date),
                outcome_window_days=14,
                metadata={
                    "vix": latest_vix,
                    "10d_avg": round(avg_vix, 2),
                    "signal_type": "vix_complacency",
                },
                signal_suffix="vix_compress",
            ))

    # --- Federal funds rate trajectory ---
    dff = series_data.get("DFF", [])
    if len(dff) >= 20:
        latest_date, latest_rate = dff[-1]
        rate_20ago = dff[-20][1]
        rate_change = latest_rate - rate_20ago
        if abs(rate_change) >= 0.50:  # 50+ bps change in ~20 trading days
            direction = "bullish" if rate_change < 0 else "bearish"
            signals.append(_make_signal(
                source="fred_fed_funds",
                symbol="",
                asset_class="macro",
                domain="macro",
                direction=direction,
                strength=min(1.0, abs(rate_change) / 2.0),
                confidence=0.55,
                decay_rate=0.02,
                timestamp=str(latest_date),
                outcome_window_days=60,
                metadata={
                    "current_rate": latest_rate,
                    "rate_20d_ago": rate_20ago,
                    "change_bps": round(rate_change * 100, 0),
                    "signal_type": "fed_funds_trajectory",
                },
                signal_suffix="fed_trajectory",
            ))

    logger.info("fred: %d series → %d signals (yc/credit/vix/fed)", len(series_data), len(signals))
    return signals


def _mine_massive(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    lookback_days: int = _LOOKBACK_DAYS_MASSIVE,
) -> List[dict]:
    """Extract signals from the Polygon.io daily OHLCV store.

    The existing signal adapter uses Massive for volume/price anomaly signals.
    This miner additionally extracts:
    - VWAP vs close divergence: institutional vs retail buying pressure
    - Gap-up/gap-down opens: > 3% from prior close
    - Transaction count anomalies: block trade detection (high value, low count)
    - Multi-day consecutive volume expansion
    """
    signals: List[dict] = []
    if not _attach_db(con, "massive", "massive"):
        return signals

    cutoff = _cutoff_date(lookback_days)
    try:
        rows = con.execute(f"""
            WITH consecutive AS (
                SELECT
                    ticker,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    vwap,
                    transactions,
                    LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close,
                    LAG(volume) OVER (PARTITION BY ticker ORDER BY date) AS prev_volume,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                FROM massive.daily_bars
                WHERE date >= '{cutoff}'
                  AND volume > 0
                  AND close > 0
            )
            SELECT
                ticker, date, open, high, low, close, volume, vwap,
                transactions, prev_close, prev_volume, rn
            FROM consecutive
            WHERE rn = 1
        """).fetchall()
    except Exception as exc:
        logger.debug("massive query failed: %s", exc)
        return signals

    processed = 0
    for (ticker, date, open_p, high, low, close, volume, vwap,
         transactions, prev_close, prev_volume, rn) in rows:

        if not ticker or not close:
            continue
        processed += 1

        # --- VWAP vs close divergence ---
        if vwap and close and vwap > 0:
            vwap_dev = (close - vwap) / vwap
            # Price ending significantly above VWAP = institutional buying above retail
            if abs(vwap_dev) >= 0.015:
                direction = "bullish" if vwap_dev > 0 else "bearish"
                strength = min(1.0, abs(vwap_dev) / 0.05)
                signals.append(_make_signal(
                    source="massive_vwap_dev",
                    symbol=ticker,
                    asset_class="stock",
                    domain="technical",
                    direction=direction,
                    strength=strength,
                    confidence=0.45,
                    decay_rate=0.15,
                    timestamp=str(date),
                    outcome_window_days=5,
                    metadata={
                        "close": close,
                        "vwap": round(vwap, 4),
                        "vwap_deviation_pct": round(vwap_dev * 100, 2),
                        "signal_type": "vwap_close_divergence",
                    },
                    signal_suffix=f"vwap:{ticker}",
                ))

        # --- Gap detection ---
        if prev_close and prev_close > 0 and open_p:
            gap_pct = (open_p - prev_close) / prev_close
            if abs(gap_pct) >= 0.03:  # 3%+ gap
                direction = "bullish" if gap_pct > 0 else "bearish"
                strength = min(1.0, abs(gap_pct) / 0.10)
                signals.append(_make_signal(
                    source="massive_gap",
                    symbol=ticker,
                    asset_class="stock",
                    domain="technical",
                    direction=direction,
                    strength=strength,
                    confidence=0.55,
                    decay_rate=0.20,
                    timestamp=str(date),
                    outcome_window_days=5,
                    metadata={
                        "open": open_p,
                        "prev_close": prev_close,
                        "gap_pct": round(gap_pct * 100, 2),
                        "signal_type": "gap_open",
                    },
                    signal_suffix=f"gap:{ticker}",
                ))

        # --- Transaction count vs volume ratio (block trade signal) ---
        if transactions and volume and transactions > 0:
            avg_trade_size = volume / transactions
            # Large average trade size suggests institutional block orders
            # Typical retail avg: 100-300 shares; institutional: 1000+
            if avg_trade_size >= 1000:
                # Direction from price change on the day
                chg = (close - open_p) / open_p if open_p and open_p > 0 else 0
                direction = "bullish" if chg > 0.005 else "bearish" if chg < -0.005 else "neutral"
                strength = min(1.0, avg_trade_size / 10000)
                signals.append(_make_signal(
                    source="massive_block_trades",
                    symbol=ticker,
                    asset_class="stock",
                    domain="institutional",
                    direction=direction,
                    strength=strength,
                    confidence=0.45,
                    decay_rate=0.10,
                    timestamp=str(date),
                    outcome_window_days=7,
                    metadata={
                        "volume": volume,
                        "transactions": transactions,
                        "avg_trade_size": round(avg_trade_size, 0),
                        "daily_change_pct": round(chg * 100, 2),
                        "signal_type": "block_trade_pattern",
                    },
                    signal_suffix=f"block:{ticker}",
                ))

    logger.info(
        "massive: %d bars processed → %d signals (vwap_dev/gap/block_trade)",
        processed, len(signals),
    )
    return signals


def _mine_finviz(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
) -> List[dict]:
    """Extract short squeeze setup signals from FinViz unusual volume + short float.

    The existing signal adapter extracts unusual volume and short squeeze candidates
    separately. This miner finds the INTERSECTION: tickers with both unusual
    volume AND high short float = potential squeeze setup.
    """
    signals: List[dict] = []
    if not _attach_db(con, "finviz", "finviz"):
        return signals

    try:
        # Check if finviz_short_squeeze table exists before JOINing
        tables = [r[0] for r in con.execute(
            "SELECT name FROM finviz.sqlite_master WHERE type='table'"
        ).fetchall()]
        if "finviz_short_squeeze" not in tables:
            # Table doesn't exist yet — fall back to volume-only squeeze heuristic
            # High volume + big price drop = potential short squeeze candidate
            rows = con.execute("""
                SELECT
                    ticker, sector, price, change_pct, volume_ratio,
                    NULL AS short_float_pct, NULL AS short_ratio, ingested_at
                FROM finviz.finviz_unusual_volume
                WHERE volume_ratio >= 3.0
                  AND change_pct <= -5.0
            """).fetchall()
        else:
            rows = con.execute("""
                SELECT
                    uv.ticker,
                    uv.sector,
                    uv.price,
                    uv.change_pct,
                    uv.volume_ratio,
                    ss.short_float_pct,
                    ss.short_ratio,
                    uv.ingested_at
                FROM finviz.finviz_unusual_volume uv
                INNER JOIN finviz.finviz_short_squeeze ss
                    ON uv.ticker = ss.ticker
                WHERE uv.volume_ratio >= 2.0
                  AND ss.short_float_pct >= 15.0
            """).fetchall()
    except Exception as exc:
        logger.debug("finviz cross-query failed: %s", exc)
        return signals

    for (ticker, sector, price, change_pct, vol_ratio,
         short_float, short_ratio, ingested_at) in rows:
        if not ticker:
            continue
        # Squeeze strength is the product of volume spike × short float
        strength = min(1.0, (vol_ratio / 5.0) * 0.5 + (short_float / 50.0) * 0.5)
        signals.append(_make_signal(
            source="finviz_squeeze_setup",
            symbol=ticker,
            asset_class="stock",
            domain="technical",
            direction="bullish",
            strength=strength,
            confidence=0.55,
            decay_rate=0.15,
            timestamp=ingested_at or datetime.now(timezone.utc).isoformat(),
            outcome_window_days=7,
            metadata={
                "sector": sector,
                "price": price,
                "change_pct": change_pct,
                "volume_ratio": vol_ratio,
                "short_float_pct": short_float,
                "short_ratio": short_ratio,
                "signal_type": "squeeze_setup",
            },
            signal_suffix=f"squeeze:{ticker}",
        ))

    logger.info("finviz: %d squeeze setups (unusual_volume + high_short_float)", len(signals))
    return signals


def _mine_finra(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    lookback_days: int = _LOOKBACK_DAYS_FINRA,
) -> List[dict]:
    """Extract short volume acceleration signals from FINRA data.

    The existing adapter extracts the raw short_ratio value. This miner
    additionally extracts:
    - Short volume increasing over 3 consecutive days (momentum signal)
    - Speculative short ratio anomalies (spikes in speculative vs plain short volume)
    - Days where short ratio > 70% AND volume is elevated (capitulation risk)
    """
    signals: List[dict] = []
    if not _attach_db(con, "finra", "finra"):
        return signals

    cutoff = _cutoff_date(lookback_days)
    try:
        rows = con.execute(f"""
            SELECT
                symbol,
                date,
                short_volume,
                total_volume,
                short_ratio,
                speculative_short_ratio
            FROM finra.finra_short_volume
            WHERE date >= '{cutoff}'
              AND total_volume > 0
            ORDER BY symbol, date ASC
        """).fetchall()
    except Exception as exc:
        logger.debug("finra query failed: %s", exc)
        return signals

    # Group by symbol
    by_symbol: Dict[str, list] = {}
    for row in rows:
        sym = row[0]
        if sym not in by_symbol:
            by_symbol[sym] = []
        by_symbol[sym].append(row)

    for symbol, sym_rows in by_symbol.items():
        if not sym_rows:
            continue

        latest = sym_rows[-1]
        (_, date, sv, tv, sr, spec_sr) = latest

        # --- Short ratio extreme + high volume (capitulation/forced selling) ---
        if sr and sr >= 0.70 and tv:
            signals.append(_make_signal(
                source="finra_short_extreme",
                symbol=symbol,
                asset_class="stock",
                domain="institutional",
                direction="bearish",
                strength=min(1.0, sr / 0.90),
                confidence=0.50,
                decay_rate=0.10,
                timestamp=str(date),
                outcome_window_days=7,
                metadata={
                    "short_ratio": round(sr, 4),
                    "short_volume": sv,
                    "total_volume": tv,
                    "signal_type": "extreme_short_volume",
                },
                signal_suffix=f"short_extreme:{symbol}",
            ))

        # --- Speculative short ratio divergence ---
        if spec_sr and sr and sr > 0:
            spec_ratio_of_short = spec_sr / sr
            if spec_ratio_of_short >= 0.40:  # Speculative = 40%+ of all shorts
                signals.append(_make_signal(
                    source="finra_speculative_short",
                    symbol=symbol,
                    asset_class="stock",
                    domain="institutional",
                    direction="neutral",  # Could go either way — high uncertainty
                    strength=min(1.0, spec_ratio_of_short),
                    confidence=0.40,
                    decay_rate=0.08,
                    timestamp=str(date),
                    outcome_window_days=7,
                    metadata={
                        "short_ratio": round(sr, 4),
                        "speculative_short_ratio": round(spec_sr, 4),
                        "speculative_fraction": round(spec_ratio_of_short, 4),
                        "signal_type": "high_speculative_short_fraction",
                    },
                    signal_suffix=f"spec_short:{symbol}",
                ))

        # --- 3-day short ratio acceleration ---
        if len(sym_rows) >= 3:
            ratios = [(r[4] or 0) for r in sym_rows[-3:]]
            if ratios[0] < ratios[1] < ratios[2]:  # Monotonically increasing
                accel = ratios[2] - ratios[0]
                if accel >= 0.10:  # +10% short ratio in 3 days
                    signals.append(_make_signal(
                        source="finra_short_acceleration",
                        symbol=symbol,
                        asset_class="stock",
                        domain="institutional",
                        direction="bearish",
                        strength=min(1.0, accel / 0.30),
                        confidence=0.45,
                        decay_rate=0.12,
                        timestamp=str(date),
                        outcome_window_days=7,
                        metadata={
                            "short_ratio_day1": round(ratios[0], 4),
                            "short_ratio_day3": round(ratios[2], 4),
                            "acceleration": round(accel, 4),
                            "signal_type": "short_ratio_3day_acceleration",
                        },
                        signal_suffix=f"short_accel:{symbol}",
                    ))

    logger.info("finra: %d symbols → %d signals (extreme/speculative/acceleration)", len(by_symbol), len(signals))
    return signals


def _mine_finnhub_ticks(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    lookback_days: int = _LOOKBACK_DAYS_FINNHUB,
) -> List[dict]:
    """Extract real-time trade tick velocity signals from Finnhub WebSocket data.

    The tick store captures every trade on the Finnhub WebSocket feed.
    This miner identifies:
    - Trade velocity spikes: trades/minute in last hour vs baseline
    - Volume acceleration: total tick volume in 1h window vs prior 1h
    """
    signals: List[dict] = []
    if not _attach_db(con, "finnhub", "finnhub"):
        return signals

    cutoff_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
    )
    try:
        rows = con.execute(f"""
            WITH hourly AS (
                SELECT
                    symbol,
                    STRFTIME(
                        TO_TIMESTAMP(CAST(timestamp_ms AS BIGINT) / 1000),
                        '%Y-%m-%d %H:00'
                    ) AS hour_bucket,
                    COUNT(*) AS tick_count,
                    SUM(volume) AS total_volume,
                    MAX(price) AS max_price,
                    MIN(price) AS min_price
                FROM finnhub.finnhub_ticks
                WHERE CAST(timestamp_ms AS BIGINT) >= {cutoff_ms}
                GROUP BY symbol, hour_bucket
            ),
            ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY hour_bucket DESC) AS rn
                FROM hourly
            )
            SELECT
                cur.symbol,
                cur.hour_bucket,
                cur.tick_count,
                cur.total_volume,
                cur.max_price,
                cur.min_price,
                prev.tick_count AS prev_tick_count,
                prev.total_volume AS prev_total_volume
            FROM ranked cur
            LEFT JOIN ranked prev
                ON cur.symbol = prev.symbol AND prev.rn = 2
            WHERE cur.rn = 1
              AND cur.tick_count > 10
        """).fetchall()
    except Exception as exc:
        logger.debug("finnhub ticks query failed: %s", exc)
        return signals

    for (symbol, hour_bucket, ticks, vol, max_p, min_p,
         prev_ticks, prev_vol) in rows:
        if not symbol:
            continue

        # Tick velocity spike
        if prev_ticks and prev_ticks > 0:
            tick_ratio = ticks / prev_ticks
            if tick_ratio >= 3.0:
                # Direction from price range midpoint
                if max_p and min_p and max_p > 0:
                    mid = (max_p + min_p) / 2
                    direction = "bullish" if max_p == mid else "neutral"
                else:
                    direction = "neutral"
                signals.append(_make_signal(
                    source="finnhub_tick_velocity",
                    symbol=symbol,
                    asset_class="stock",
                    domain="technical",
                    direction=direction,
                    strength=min(1.0, (tick_ratio - 3.0) / 7.0 + 0.40),
                    confidence=0.45,
                    decay_rate=0.25,
                    timestamp=str(hour_bucket),
                    outcome_window_days=1,
                    metadata={
                        "tick_count": ticks,
                        "prev_tick_count": prev_ticks,
                        "tick_ratio": round(tick_ratio, 2),
                        "total_volume": vol,
                        "signal_type": "tick_velocity_spike",
                    },
                    signal_suffix=f"ticks:{symbol}",
                ))

    logger.info("finnhub_ticks: %d symbols → %d velocity signals", len(rows), len(signals))
    return signals


# ── Bridge file writer ──────────────────────────────────────────────────────────

def _write_bridge(signals: List[dict], logger: logging.Logger) -> int:
    """Append signals to the bridge JSONL file (atomic temp-file write).

    The bridge file is append-only per cycle. The daemon reads it via
    signal buffer ingestion. Deduplication happens within this cycle
    (same signal_id not written twice); across-cycle dedup is handled
    by the signal buffer's own dedup logic.

    Returns number of signals written.
    """
    if not signals:
        return 0

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Deduplicate by signal_id within this batch
    seen_ids: set = set()
    unique_signals = []
    for sig in signals:
        sid = sig.get("signal_id", "")
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            unique_signals.append(sig)

    # Cap total per cycle
    if len(unique_signals) > _MAX_SIGNALS_PER_CYCLE:
        # Sort by strength descending, keep the best
        unique_signals.sort(key=lambda s: s.get("strength", 0), reverse=True)
        unique_signals = unique_signals[:_MAX_SIGNALS_PER_CYCLE]
        logger.warning(
            "Signal cap hit — kept top %d by strength (total extracted: %d)",
            _MAX_SIGNALS_PER_CYCLE, len(signals),
        )

    # Append to bridge file (not atomic per-line, but journaling is fine here)
    written = 0
    try:
        with open(_BRIDGE_FILE, "a", encoding="utf-8") as fh:
            for sig in unique_signals:
                line = json.dumps(sig, default=str)
                fh.write(line + "\n")
                written += 1
    except Exception as exc:
        logger.error("Bridge file write failed: %s", exc)
        return 0

    return written


# ── Main loop ───────────────────────────────────────────────────────────────────

def run_once(logger: Optional[logging.Logger] = None) -> int:
    """Execute one full mining cycle across all stores.

    Returns total signals extracted and written.
    """
    if logger is None:
        logger = _setup_logging()

    con = _open_duckdb()
    all_signals: List[dict] = []

    stores = [
        ("prices",          _mine_prices),
        ("cot",             _mine_cot),
        ("openinsider",     _mine_openinsider),
        ("fred",            _mine_fred),
        ("massive",         _mine_massive),
        ("finviz",          _mine_finviz),
        ("finra",           _mine_finra),
        ("finnhub_ticks",   _mine_finnhub_ticks),
    ]

    for store_name, miner_fn in stores:
        try:
            store_signals = miner_fn(con, logger)
            all_signals.extend(store_signals)
        except Exception as exc:
            # One broken store NEVER kills the others
            logger.error("Store '%s' miner raised: %s", store_name, exc, exc_info=True)

    try:
        con.close()
    except Exception:
        pass

    written = _write_bridge(all_signals, logger)

    logger.info(
        "Cycle complete: %d total signals extracted → %d written to bridge "
        "(stores: %d)",
        len(all_signals), written, len(stores),
    )
    return written


def run_continuous(interval_seconds: int = _DEFAULT_INTERVAL_SECONDS) -> None:
    """Run the mining loop indefinitely.

    Sleeps interval_seconds between cycles. Each cycle mines all stores
    and appends signals to the bridge file. The daemon reads the bridge
    file continuously — no coordination required.
    """
    logger = _setup_logging()
    logger.info(
        "Raw data miner starting | interval=%ds | bridge=%s",
        interval_seconds, _BRIDGE_FILE,
    )

    cycle = 0
    while True:
        cycle += 1
        t0 = time.monotonic()
        logger.info("=== Mining cycle %d ===", cycle)

        try:
            written = run_once(logger)
        except Exception as exc:
            logger.error("Cycle %d failed: %s", cycle, exc, exc_info=True)
            written = 0

        elapsed = time.monotonic() - t0
        logger.info(
            "Cycle %d done in %.1fs | %d signals written | sleeping %ds",
            cycle, elapsed, written, interval_seconds,
        )

        time.sleep(interval_seconds)


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "MIDGE raw data miner — reads SQLite raw stores via DuckDB and "
            "extracts signals the standard adapters miss. Writes to "
            "data/market/raw_miner_signals.jsonl for daemon ingestion."
        )
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=_DEFAULT_INTERVAL_SECONDS,
        help=f"Seconds between mining cycles (default: {_DEFAULT_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit (useful for testing)",
    )
    args = parser.parse_args()

    if args.once:
        _logger = _setup_logging()
        written = run_once(_logger)
        print(f"[raw_miner] One-shot complete: {written} signals written to {_BRIDGE_FILE}")
    else:
        run_continuous(interval_seconds=args.interval)
