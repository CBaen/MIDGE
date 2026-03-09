#!/usr/bin/env python3
"""
Convergence Alerter - Generates alerts when patterns converge across domains.

CORE OUTPUT for trading decisions:
- Detects when multiple signals from different domains align
- "crypto whale + congress trade + hiring surge = actionable signal"
- Combines velocity, correlation, and confidence into single alert

Usage:
    from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter

    alerter = ConvergenceAlerter()
    alerter.record_signal("insider_buys", 0.8, "insider", direction="bullish")
    alerter.record_signal("crypto_whales", 0.7, "crypto", direction="bullish")

    alerts = alerter.check_convergence()
    for alert in alerts:
        print(alert.summary)
"""

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Discovery log path — same resolution as thompson_sampler.py
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
_DISCOVERY_LOG = _DATA_DIR / "discovery_log.jsonl"

# Independence correction thresholds — controls how correlated domains
# are discounted in the effective domain count calculation.
# Phase 0 found macro+technical at r=0.73 — these thresholds gate that fix.
STRONG_CORRELATION_THRESHOLD = 0.5   # |r| above this → half credit (+0.5)
MODERATE_CORRELATION_THRESHOLD = 0.3  # |r| above this → partial credit (+0.7)


@dataclass
class Signal:
    """Single signal observation."""
    signal_id: str
    strength: float  # 0-1 normalized
    domain: str
    direction: str  # bullish, bearish, neutral
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    velocity: float = 0.0  # Rate of change
    confidence: float = 0.5  # Reliability estimate
    source: str = ""  # Original source type for Thompson key lookup


@dataclass
class ConvergenceAlert:
    """Alert generated when multiple domains converge."""
    alert_id: str
    timestamp: datetime
    direction: str  # bullish, bearish
    strength: float  # Overall convergence strength 0-1
    confidence: float  # Reliability estimate 0-1
    domains_converging: List[str]
    signals: List[Signal]
    cross_domain_count: int
    summary: str
    urgency: str  # immediate, hours, days
    # Coherence fields — Capability 3 (narrative conflict detection).
    # Defaults preserve backward compatibility with existing callers.
    coherence: float = 1.0            # 1.0 = all signals agree, 0.5 = evenly split
    contradiction_details: list = field(default_factory=list)  # [(domain, direction), ...]
    combo_key: str = ""  # Thompson key for this domain combination (e.g. "combo:events+macro+price")
    # Temporal ordering fields — domains sorted by when they fired (earliest first).
    # sequence_score > 1.0 = ordering matches known lag relationships (boost).
    # sequence_score < 1.0 = ordering reversed vs. known lags (discount).
    # Default 1.0 = neutral (no lag data or single domain).
    domain_sequence: List[str] = field(default_factory=list)
    sequence_score: float = 1.0
    # Causal cascade — WorldModel ripple effects predicting downstream dominoes.
    # Each entry: {ticker, direction, strength, lag_days, path, confidence}
    ripple_effects: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "strength": round(self.strength, 3),
            "confidence": round(self.confidence, 3),
            "domains": self.domains_converging,
            "signal_count": len(self.signals),
            "cross_domain_count": self.cross_domain_count,
            "summary": self.summary,
            "urgency": self.urgency,
            "coherence": round(self.coherence, 3),
            "contradiction_details": self.contradiction_details,
            "combo_key": self.combo_key,
            "domain_sequence": self.domain_sequence,
            "sequence_score": round(self.sequence_score, 3),
            "ripple_effects": self.ripple_effects,
        }


class ConvergenceAlerter:
    """
    Detects convergence across domains and generates actionable alerts.

    Key insight: Single-domain signals are noisy. When 3+ domains
    from different categories all point the same direction, that's
    a much stronger signal.

    Example convergence:
    - Insider domain: Executives buying (bullish)
    - Crypto domain: Whales accumulating (bullish)
    - Government domain: Contract awarded (bullish)
    → CONVERGENCE ALERT: 3 domains bullish = high confidence

    Confidence is computed via Thompson-weighted geometric mean when
    a ThompsonSampler is provided. This replaces the original additive
    formula (Alpha's standing dissent — see deliverable.md).
    """

    # Sweep sources eligible for granular Thompson cascade (Bridge 2).
    _SWEEP_SOURCES = {"session_sweep", "session_sweep_ifvg"}

    # Maps MarketSignal.source values -> ThompsonSampler distribution keys.
    # signal.py uses descriptive source names, Thompson uses shorter keys
    # from learning_config.py source_reliability dict.
    _SOURCE_TO_THOMPSON_KEY = {
        "sec_form4": "sec_form4",
        "sec_form8k": "sec_form8k",
        "sec_efts": "sec_efts",
        "congressional": "congressional",
        "senate": "senate",
        "insider_cluster": "insider_cluster",
        "politician_correlation": "congressional",
        "contract_award": "contract_award",
        "contract_prediction": "contract_prediction",
        "hiring_tracker": "hiring_tracker",
        "sam_gov": "sam_gov",
        "social_sentiment": "social_sentiment",
        "finra_short": "finra_short",
        "finnhub_news": "finnhub_news",
        "finnhub_earnings": "finnhub_earnings",
        "fred_macro": "fred_macro",
        "session_sweep": "session_sweep",
        "session_sweep_ifvg": "session_sweep_ifvg",
        "ta_rsi": "ta_rsi",
        "ta_macd": "ta_macd",
        "ta_bollinger": "ta_bollinger",
        "ta_structure": "ta_structure",
        "ta_candle": "ta_candle",
        # Ten Gifts: Wave 1
        "order_flow": "order_flow",
        # Ten Gifts: Wave 2
        "fractal_resonance": "fractal_resonance",
        "archetype_match": "archetype_match",
        # New sources (Layer 6)
        "cot_positioning": "cot_positioning",
        "stocktwits_sentiment": "stocktwits_sentiment",
        "vix_term_structure": "vix_term_structure",
        "google_trends": "google_trends",
        "finnhub_economic": "finnhub_economic",
        "finnhub_analyst": "finnhub_analyst",
        "finnhub_earnings_calendar": "finnhub_earnings_calendar",
        # Real-economy: Energy
        "eia_energy": "eia_energy",
        # Real-economy: Legislative
        "congress_legislation": "congress_legislation",
        # Always-On Wave 2: Real-Time + Crypto
        "finnhub_realtime": "finnhub_realtime",
        "crypto_coingecko": "crypto_coingecko",
        "crypto_coincap": "crypto_coincap",
        # Always-On Wave 3: Data Enrichment
        "openinsider_purchase": "openinsider_purchase",
        "institutional_13f": "institutional_13f",
        "activist_13d": "activist_13d",
        "finviz_unusual_volume": "finviz_unusual_volume",
        "finviz_short_squeeze": "finviz_short_squeeze",
        "economic_calendar": "economic_calendar",
        # Massive/Polygon.io
        "massive_snapshot": "massive_snapshot",
        # Price data
        "yfinance_price": "yfinance_price",
        # Pattern discovery (motif + anomaly detectors)
        "motif_match": "motif_match",
        "price_discord": "price_discord",
        "streaming_anomaly": "streaming_anomaly",
    }

    # Domain → list of source names used in lag_correlations.json.
    # Enables domain-level correlation lookup from source-level correlation data.
    _DOMAIN_SOURCES: Dict[str, List[str]] = {
        "insider": ["sec_form4", "openinsider_purchase", "insider_cluster"],
        "events": ["sec_form8k", "sec_efts", "finnhub_earnings", "finnhub_news",
                   "finnhub_realtime", "finnhub_earnings_calendar", "massive_snapshot",
                   "hiring_tracker"],
        "macro": ["fred_macro", "economic_calendar"],
        "technical": ["ta_rsi", "ta_macd", "ta_bollinger", "ta_structure", "ta_candle",
                      "session_sweep", "session_sweep_ifvg", "fractal_resonance",
                      "order_flow", "finviz_unusual_volume", "finviz_short_squeeze",
                      "yfinance_price", "motif_match", "price_discord",
                      "streaming_anomaly"],
        "sentiment": ["social_sentiment", "google_trends", "stocktwits_sentiment"],
        "government": ["congressional", "senate", "congress_legislation"],
        "contracts": ["contract_award", "contract_prediction", "sam_gov"],
        "fundamentals": ["finnhub_analyst"],
        "positioning": ["cot_positioning"],
        "volatility": ["vix_term_structure"],
        "crypto": ["crypto_coingecko", "crypto_coincap"],
        "institutional": ["activist_13d", "institutional_13f", "finra_short"],
        "energy": ["eia_energy"],
    }

    def __init__(
        self,
        min_domains: int = 3,
        min_strength: float = 0.6,
        convergence_window_hours: int = 72,
        persistence_path: str = None,
        thompson_sampler=None,
        regime_classifier=None,
        causal_engine=None,
        event_bus=None,
        catalyst_calendar=None,
        cross_asset_confirmer=None,
        deception_detector=None,
        pattern_archetype_engine=None,
        economic_calendar=None,
        correlation_tracker=None,
        world_model=None,
        quorum_space=None,
    ):
        """
        Initialize convergence alerter.

        Args:
            min_domains: Minimum different domains to trigger alert
            min_strength: Minimum average signal strength
            convergence_window_hours: How recent signals must be
            persistence_path: Path for alert history
            thompson_sampler: Optional ThompsonSampler for reliability weights
            regime_classifier: Optional RegimeClassifier for regime-aware Thompson queries
            causal_engine: Optional CausalReasoningEngine for contradiction analysis
            event_bus: Optional EventBus for publishing CH_CONTRADICTION_DETECTED
            catalyst_calendar: Optional CatalystCalendar for timing context modifiers
            cross_asset_confirmer: Optional CrossAssetConfirmer for cross-asset confirmation
            deception_detector: Optional DeceptionDetector for manipulation detection
            pattern_archetype_engine: Optional PatternArchetypeEngine for archetype context
            economic_calendar: Optional EconomicCalendar for suppression windows
            correlation_tracker: Optional CorrelationTracker for domain independence scoring
            quorum_space: Optional QuorumSpace for multi-agent collective confidence boost
        """
        self.min_domains = min_domains
        self.min_strength = min_strength
        self.convergence_window = timedelta(hours=convergence_window_hours)
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._thompson = thompson_sampler
        self._regime_classifier = regime_classifier
        self._causal_engine = causal_engine
        self._world_model = world_model
        self._bus = event_bus
        self._catalyst_calendar = catalyst_calendar
        self._cross_asset_confirmer = cross_asset_confirmer
        self._deception_detector = deception_detector
        self._pattern_archetype_engine = pattern_archetype_engine
        self._economic_calendar = economic_calendar
        self._correlation_tracker = correlation_tracker
        self._quorum_space = quorum_space
        self._cached_regime = ("default", 0.0)  # (regime_str, timestamp)

        # Per-domain convergence windows — slow-moving data sources need longer lookback.
        # Domains NOT listed here fall back to self.convergence_window (72h global default).
        self._domain_windows: Dict[str, timedelta] = {
            "positioning": timedelta(hours=14 * 24),  # 14 days — COT is weekly data
            "government": timedelta(hours=7 * 24),    # 7 days — congressional trades
            "contracts": timedelta(hours=7 * 24),     # 7 days — contract awards
            "energy": timedelta(hours=7 * 24),        # 7 days — weekly EIA reports
        }

        # Recent signals by domain
        self.signals: Dict[str, List[Signal]] = defaultdict(list)

        # Alert history
        self.alerts: List[ConvergenceAlert] = []

        # Domain categories for cross-domain verification
        self.domain_categories = {
            "insider": "behavioral",
            "congress": "behavioral",
            "crypto": "market",
            "technical": "market",
            "volume": "market",
            "sentiment": "social",
            "reddit": "social",
            "news": "information",
            "events": "information",
            "fundamentals": "financial",
            "macro": "financial",
            "government": "institutional",
            "contracts": "institutional",
            "institutional_synthesis": "institutional",
            # New domains (Layer 6)
            "positioning": "institutional",
            "volatility": "market",
            # Real-economy
            "energy": "fundamental",
        }

        self._alert_counter = 0

        # Alert deduplication (defense in depth — step hook also deduplicates)
        # Per-direction tracking: both bullish and bearish can be suppressed
        # independently, preventing the cross-evasion bug where each direction
        # kept overwriting the other's suppression record.
        self._last_alert_times: Dict[str, datetime] = {}
        self._alert_lock = threading.Lock()
        self._min_alert_interval_hours = 4.0

        # Lag findings for temporal ordering (Task 1 + Task 3).
        # Set via set_lag_findings() after LagCorrelationAnalyzer runs.
        # Loaded from disk on startup so alerter has data from prior runs.
        self._lag_findings: List = []
        self._load_lag_findings()

    # ------------------------------------------------------------------
    # Lag findings integration (Task 1 + Task 3)
    # ------------------------------------------------------------------

    def set_lag_findings(self, findings: List) -> None:
        """Accept lag findings from LagCorrelationAnalyzer.

        Called by the step hook after every lag analysis run (every 500 steps).
        Findings shape the sequence_score on future alerts.

        Args:
            findings: List of LagFinding objects (source_a leads source_b by lag_days)
        """
        self._lag_findings = findings if findings else []
        logger.info(
            "ConvergenceAlerter: received %d lag findings for sequence scoring",
            len(self._lag_findings),
        )

    def _load_lag_findings(self) -> None:
        """Load persisted lag findings on startup so sequence scoring works immediately."""
        lag_path = _DATA_DIR / "lag_correlations.json"
        if not lag_path.exists():
            return
        try:
            from mae_core.market.intelligence.lag_correlation_analyzer import LagFinding
            raw = json.loads(lag_path.read_text(encoding="utf-8"))
            self._lag_findings = [LagFinding(**r) for r in raw]
            logger.debug(
                "ConvergenceAlerter: loaded %d lag findings from disk",
                len(self._lag_findings),
            )
        except Exception:
            logger.debug("ConvergenceAlerter: failed to load lag findings", exc_info=True)
            self._lag_findings = []

    def _build_domain_sequence(self, signals: list) -> List[str]:
        """Sort contributing signals by timestamp (earliest first) and return domain order.

        Returns a list of domain names ordered from first-to-fire to last-to-fire.
        Domains with multiple signals use the timestamp of the earliest signal.
        """
        try:
            domain_first_ts: Dict[str, datetime] = {}
            for sig in signals:
                ts = getattr(sig, "timestamp", None)
                if ts is None:
                    continue
                domain = getattr(sig, "domain", "")
                if domain and (domain not in domain_first_ts or ts < domain_first_ts[domain]):
                    domain_first_ts[domain] = ts
            if not domain_first_ts:
                return []
            return [d for d, _ in sorted(domain_first_ts.items(), key=lambda x: x[1])]
        except Exception:
            logger.debug("domain sequence build failed", exc_info=True)
            return []

    def _compute_sequence_score(self, domain_sequence: List[str]) -> float:
        """Score how well the observed domain firing order matches known lag relationships.

        For each pair (A, B) in domain_sequence where A fired before B:
        - If LagFinding says A leads B → matches → contributes a boost (+).
        - If LagFinding says B leads A → reversed → contributes a discount (-).
        - No data → neutral.

        Score formula:
            score = 1.0 + sum(matched_boosts) - sum(reversed_penalties)
        Clamped to [0.5, 1.3] so it never destroys or artificially doubles confidence.

        Returns:
            float in [0.5, 1.3]. 1.0 = neutral (no data or no matches).
        """
        if len(domain_sequence) < 2 or not self._lag_findings:
            return 1.0

        try:
            # Build a domain-level lag lookup: domain_a leads domain_b → lag_days
            # Uses _DOMAIN_SOURCES to map sources → domains.
            # Format: (leading_domain, lagging_domain) → max |correlation|
            domain_lead: Dict = {}  # (lead_domain, lag_domain) → strength
            domain_lag_reverse: Dict = {}  # (lag_domain, lead_domain) → strength

            for finding in self._lag_findings:
                src_a = finding.source_a
                src_b = finding.source_b
                corr = abs(finding.correlation)
                # Find which domains these sources belong to
                dom_a = self._source_to_domain(src_a)
                dom_b = self._source_to_domain(src_b)
                if dom_a and dom_b and dom_a != dom_b:
                    key = (dom_a, dom_b)
                    # source_a leads source_b (A fires first, B follows)
                    if corr > domain_lead.get(key, 0.0):
                        domain_lead[key] = corr

            score_delta = 0.0
            pair_count = 0

            for i, dom_a in enumerate(domain_sequence):
                for dom_b in domain_sequence[i + 1:]:
                    # dom_a fired before dom_b in this alert
                    pair_count += 1
                    forward_key = (dom_a, dom_b)   # matches lag data
                    reverse_key = (dom_b, dom_a)   # contradicts lag data

                    forward_strength = domain_lead.get(forward_key, 0.0)
                    reverse_strength = domain_lead.get(reverse_key, 0.0)

                    if forward_strength > 0.3:
                        # Alert ordering matches known lag → slight boost
                        score_delta += 0.05 * forward_strength
                    elif reverse_strength > 0.3:
                        # Alert ordering REVERSES known lag → slight discount
                        score_delta -= 0.05 * reverse_strength

            if pair_count == 0:
                return 1.0

            raw_score = 1.0 + score_delta
            return max(0.5, min(1.3, raw_score))

        except Exception:
            logger.debug("sequence score computation failed", exc_info=True)
            return 1.0

    def _source_to_domain(self, source: str) -> str:
        """Map a source name to its domain using _DOMAIN_SOURCES (reverse lookup)."""
        for domain, sources in self._DOMAIN_SOURCES.items():
            if source in sources:
                return domain
        return ""

    # ------------------------------------------------------------------
    # Signal buffer persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _json_default(obj):
        """JSON serializer for objects not handled by default encoder."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        if isinstance(obj, set):
            return list(obj)
        return str(obj)

    def save_signal_buffer(self) -> int:
        """Persist in-memory signal buffer and alerter state to disk.

        Saves only signals that are still within their domain convergence
        window — expired signals are not worth carrying across restarts.
        Uses atomic writes (write to .tmp then rename) for thread safety,
        matching the pattern used by ThompsonSampler.

        Returns the number of signals saved (0 if buffer was empty).
        """
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        total_in_memory = sum(len(v) for v in self.signals.values())
        logger.info(
            "Signal buffer save starting: %d signals across %d domains in memory",
            total_in_memory,
            len(self.signals),
        )

        # Build serialisable buffer — only keep live (non-expired) signals.
        buffer: Dict[str, list] = {}
        for domain, sigs in list(self.signals.items()):
            window = self._domain_windows.get(domain, self.convergence_window)
            cutoff = now - window
            live = []
            for s in sigs:
                try:
                    if s.timestamp >= cutoff:
                        live.append({
                            "signal_id": s.signal_id,
                            "strength": s.strength,
                            "domain": s.domain,
                            "direction": s.direction,
                            "timestamp": s.timestamp.isoformat(),
                            "metadata": s.metadata,
                            "velocity": s.velocity,
                            "confidence": s.confidence,
                            "source": s.source,
                        })
                except Exception:
                    logger.debug("Skipping non-serializable signal in %s", domain)
            if live:
                buffer[domain] = live

        # Atomically write signal buffer.
        buf_path = _DATA_DIR / "signal_buffer.json"
        tmp_buf = buf_path.with_suffix(".tmp")
        try:
            tmp_buf.write_text(json.dumps(buffer, indent=2, default=self._json_default))
            os.replace(tmp_buf, buf_path)
        except Exception:
            logger.warning("Failed to save signal buffer", exc_info=True)
            if tmp_buf.exists():
                tmp_buf.unlink(missing_ok=True)
            return 0

        # Atomically write alerter state (counter + dedup times).
        state = {
            "alert_counter": self._alert_counter,
            "last_alert_times": {
                k: (v.isoformat() if isinstance(v, datetime) else str(v))
                for k, v in self._last_alert_times.items()
            },
        }
        state_path = _DATA_DIR / "alerter_state.json"
        tmp_state = state_path.with_suffix(".tmp")
        try:
            tmp_state.write_text(json.dumps(state, indent=2))
            os.replace(tmp_state, state_path)
        except Exception:
            logger.warning("Failed to save alerter state", exc_info=True)
            if tmp_state.exists():
                tmp_state.unlink(missing_ok=True)
            return 0

        total_signals = sum(len(v) for v in buffer.values())
        logger.info(
            "Signal buffer saved: %d signals across %d domains",
            total_signals,
            len(buffer),
        )
        return total_signals

    def load_signal_buffer(self) -> int:
        """Restore signal buffer and alerter state from disk.

        Signals older than their domain convergence window are discarded
        on load — only signals that are still relevant are restored.
        Missing or corrupt files are handled gracefully.

        Returns the number of signals restored (0 if none found).
        """
        now = datetime.now()

        # Restore signal buffer.
        buf_path = _DATA_DIR / "signal_buffer.json"
        if buf_path.exists():
            try:
                raw: Dict[str, list] = json.loads(buf_path.read_text())
                loaded = 0
                for domain, entries in raw.items():
                    window = self._domain_windows.get(domain, self.convergence_window)
                    cutoff = now - window
                    for entry in entries:
                        ts = datetime.fromisoformat(entry["timestamp"])
                        if ts < cutoff:
                            continue  # expired — skip
                        self.signals[domain].append(
                            Signal(
                                signal_id=entry["signal_id"],
                                strength=entry["strength"],
                                domain=entry["domain"],
                                direction=entry["direction"],
                                timestamp=ts,
                                metadata=entry.get("metadata", {}),
                                velocity=entry.get("velocity", 0.0),
                                confidence=entry.get("confidence", 0.5),
                                source=entry.get("source", ""),
                            )
                        )
                        loaded += 1
                domain_count = len(self.signals)
                logger.info(
                    "Signal buffer loaded: %d signals across %d domains",
                    loaded,
                    domain_count,
                )
            except Exception:
                logger.warning("Failed to load signal buffer — starting fresh", exc_info=True)
                loaded = 0
        else:
            logger.debug("No signal buffer file found — starting with empty buffer")
            loaded = 0

        # Restore alerter state.
        state_path = _DATA_DIR / "alerter_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                self._alert_counter = int(state.get("alert_counter", 0))
                raw_times = state.get("last_alert_times", {})
                self._last_alert_times = {
                    k: datetime.fromisoformat(v) for k, v in raw_times.items()
                }
                logger.debug(
                    "Alerter state loaded: counter=%d, dedup_entries=%d",
                    self._alert_counter,
                    len(self._last_alert_times),
                )
            except Exception:
                logger.warning("Failed to load alerter state — using defaults", exc_info=True)
        else:
            logger.debug("No alerter state file found — using defaults")

        return loaded

    def record_signal(
        self,
        signal_id: str,
        strength: float,
        domain: str,
        direction: str = "neutral",
        confidence: float = 0.5,
        velocity: float = 0.0,
        timestamp: datetime = None,
        metadata: dict = None,
        source: str = "",
    ):
        """
        Record a signal observation.

        Args:
            signal_id: Unique signal identifier
            strength: Signal strength 0-1 (higher = stronger)
            domain: Domain category (insider, crypto, sentiment, etc.)
            direction: bullish, bearish, or neutral
            confidence: Reliability estimate 0-1
            velocity: Rate of change (from VelocityDetector)
            timestamp: When observed
            metadata: Additional context
            source: Original source type (e.g. "sec_form4") for Thompson lookup
        """
        # Input validation — clamp to valid ranges
        strength = max(0.0, min(1.0, strength))
        confidence = max(0.0, min(1.0, confidence))
        if direction not in ("bullish", "bearish", "neutral"):
            direction = "neutral"

        timestamp = timestamp or datetime.now()
        metadata = metadata or {}

        signal = Signal(
            signal_id=signal_id,
            strength=strength,
            domain=domain,
            direction=direction,
            timestamp=timestamp,
            metadata=metadata,
            velocity=velocity,
            confidence=confidence,
            source=source,
        )

        self.signals[domain].append(signal)

        # Prune old signals
        self._prune_old_signals()

    def _prune_old_signals(self):
        """Remove signals outside the convergence window.

        Each domain may have its own extended window (self._domain_windows).
        Domains not in the dict use the global convergence_window (72h default).
        Domains whose signal list empties after pruning are removed entirely to
        keep self.signals clean.
        """
        now = datetime.now()
        for domain in list(self.signals.keys()):
            window = self._domain_windows.get(domain, self.convergence_window)
            cutoff = now - window
            self.signals[domain] = [
                s for s in self.signals[domain]
                if s.timestamp >= cutoff
            ]
            if not self.signals[domain]:
                del self.signals[domain]

    def _get_regime(self) -> str:
        """Get current market regime with 60s cache to avoid repeated calls."""
        if self._regime_classifier is None:
            return "default"
        now = time.monotonic()
        regime, cached_at = self._cached_regime
        if now - cached_at < 60.0:
            return regime
        try:
            regime = self._regime_classifier.classify()
        except Exception:
            regime = "default"
        self._cached_regime = (regime, now)
        return regime

    def _resolve_thompson_key(self, signal: Signal) -> str:
        """Resolve the Thompson distribution key for a signal.

        For sweep sources, runs a most-specific-wins cascade through
        granular backtest-derived keys (Bridge 2):
          sweep_bt:{symbol}:{direction} → sweep_bt:{symbol} →
          sweep_bt:{direction} → generic session_sweep fallback.

        For all other sources with metadata (Capability 2 — contextual Thompson):
          {source}:{role}:{sector} → {source}:{role} → {source} fallback.
          The "size" dimension (large/small) is appended when role is present:
          {source}:{role}:{size} → {source}:{role} as the final intermediate.
          Size is derived: "large" if transaction_value > $500K, else "small".

        Only selects a granular key if it has >= 5 samples (mature data).
        Final fallback: static _SOURCE_TO_THOMPSON_KEY map (existing behavior).
        """
        # --- Sweep sources: Bridge 2 cascade ---
        if signal.source in self._SWEEP_SOURCES:
            if self._thompson is None:
                return self._SOURCE_TO_THOMPSON_KEY.get(
                    signal.source, signal.source or "unknown"
                )

            symbol = signal.metadata.get("symbol", "")
            direction = signal.direction or ""
            regime = self._get_regime()

            candidates = []
            if symbol and direction:
                candidates.append(f"sweep_bt:{symbol}:{direction}")
            if symbol:
                candidates.append(f"sweep_bt:{symbol}")
            if direction:
                candidates.append(f"sweep_bt:{direction}")

            for key in candidates:
                try:
                    dist = self._thompson.get_distribution(key, regime)
                    if dist.samples >= 5:
                        logger.debug("Thompson cascade: %s → %s (n=%d)",
                                     signal.source, key, dist.samples)
                        return key
                except Exception:
                    continue

            return self._SOURCE_TO_THOMPSON_KEY.get(
                signal.source, signal.source or "unknown"
            )

        # --- All other sources: contextual cascade (Capability 2) ---
        if self._thompson is not None and signal.metadata:
            base = self._SOURCE_TO_THOMPSON_KEY.get(
                signal.source, signal.source or "unknown"
            )
            role = signal.metadata.get("role", "")
            sector = signal.metadata.get("sector", "")
            tx_value = signal.metadata.get("transaction_value", 0)
            size = "large" if tx_value > 500_000 else "small" if tx_value > 0 else ""

            regime = self._get_regime()

            # Build candidates most-specific → least-specific
            contextual_candidates = []
            if role and sector and size:
                contextual_candidates.append(f"{base}:{role}:{sector}:{size}")
            if role and sector:
                contextual_candidates.append(f"{base}:{role}:{sector}")
            if role and size:
                contextual_candidates.append(f"{base}:{role}:{size}")
            if role:
                contextual_candidates.append(f"{base}:{role}")

            for key in contextual_candidates:
                try:
                    dist = self._thompson.get_distribution(key, regime)
                    if dist.samples >= 5:
                        logger.debug("Contextual Thompson cascade: %s → %s (n=%d)",
                                     signal.source, key, dist.samples)
                        return key
                except Exception:
                    continue

        # --- Static map fallback (existing behavior) ---
        return self._SOURCE_TO_THOMPSON_KEY.get(
            signal.source, signal.source or "unknown"
        )

    def _get_thompson_weight(self, signal: Signal, regime: str) -> float:
        """Return Thompson-blended reliability weight for a signal.

        Returns a weight in [0.5, 1.5]:
        - 1.0 = neutral (trust signal's own confidence as-is)
        - >1.0 = Thompson says this source is more reliable than average
        - <1.0 = Thompson says this source is less reliable than average

        When Thompson is not configured or has no data, returns 1.0 (neutral).
        """
        if self._thompson is None:
            return 1.0

        thompson_key = self._resolve_thompson_key(signal)

        try:
            dist = self._thompson.get_distribution(thompson_key, regime)
            observations = dist.samples  # alpha + beta - 2

            if observations < 5:
                # Thin data: blend toward 1.0 (neutral) proportional to obs count
                blend = observations / 5.0
                raw_weight = 0.5 + dist.mean  # map [0,1] -> [0.5, 1.5]
                return 1.0 * (1.0 - blend) + raw_weight * blend
            else:
                # Mature data: use Thompson posterior mean as weight
                return 0.5 + dist.mean
        except Exception:
            return 1.0

    def _compute_freshness(self, signal: Signal, domain: str) -> float:
        """Compute temporal freshness weight for a signal (Capability 5).

        Recent signals are weighted more heavily than stale ones within
        their convergence window.  Uses sqrt decay (concave curve) —
        aggressive early dropoff so yesterday's signal is noticeably
        dimmer, but a 0.3 floor preserves domain presence.

        Examples (72h global window):
          2h old  → 0.83
          24h old → 0.42
          70h old → 0.30 (floored)
        """
        now = datetime.now()
        age_hours = max(0, (now - signal.timestamp).total_seconds() / 3600)
        window = self._domain_windows.get(domain, self.convergence_window)
        window_hours = window.total_seconds() / 3600

        if window_hours <= 0:
            return 1.0

        freshness = 1.0 - (age_hours / window_hours) ** 0.5
        return max(0.3, freshness)

    def _apply_quorum_boost(
        self,
        confidence: float,
        ticker: str,
        direction: str,
    ) -> float:
        """Apply quorum contributor count as a confidence multiplier.

        When 3+ independent agents all deposit signals for the same
        ticker+direction in the QuorumSpace, that collective agreement
        boosts our confidence in the convergence.

        Multiplier schedule:
            count < 3:  1.0x (no boost)
            count == 3: 1.1x
            count == 4: 1.2x
            count >= 5: 1.3x (capped)

        Final confidence is capped at 1.0.

        Args:
            confidence: Current confidence value before quorum adjustment.
            ticker: Instrument symbol (e.g. "AAPL", "CL=F").
            direction: "bullish" or "bearish".

        Returns:
            Adjusted confidence value in [0.0, 1.0].
        """
        if self._quorum_space is None or not ticker:
            return confidence
        try:
            signal_key = f"{direction}:{ticker}"
            count = self._quorum_space.get_contributor_count(signal_key)
            if count < 3:
                return confidence
            multiplier = 1.0 + min(0.3, (count - 2) * 0.1)
            boosted = min(1.0, confidence * multiplier)
            logger.debug(
                "Quorum boost: %d contributors on %s:%s → %.2f multiplier",
                count,
                direction,
                ticker,
                multiplier,
            )
            return boosted
        except Exception:
            logger.debug("Quorum boost failed gracefully", exc_info=True)
            return confidence

    def _compute_confidence(
        self,
        signals: list,
        cross_domain_count: int,
    ) -> float:
        """Compute convergence confidence via Thompson-weighted geometric mean.

        Replaces the additive formula (Alpha's standing dissent). The geometric
        mean correctly handles correlated signals without blowing up confidence
        the way additive or log-odds combination does.

        With thin Thompson data, weights are ~1.0 so the result approximates
        arithmetic mean. As Thompson accumulates outcomes, unreliable sources
        are down-weighted and reliable sources are up-weighted automatically.

        The diversity bonus now uses an effective domain count that discounts
        correlated domains (e.g. institutional+macro at r=0.57 count as ~1.5 domains,
        not 2). When CorrelationTracker is not available, falls back to the raw
        domain count for backward compatibility.
        """
        if not signals:
            return 0.5

        regime = self._get_regime()

        log_sum = 0.0
        weight_sum = 0.0

        for sig in signals:
            # Clamp confidence to prevent log(0)
            c = max(0.01, min(0.99, sig.confidence))
            w = self._get_thompson_weight(sig, regime)

            log_sum += w * math.log(c)
            weight_sum += w

        # Weighted geometric mean of per-signal confidences
        geo_mean = math.exp(log_sum / weight_sum) if weight_sum > 0 else 0.5

        # Compute the effective number of independent domains.
        # Correlated domains (e.g. institutional+macro r=0.57) receive partial credit.
        # Falls back to raw cross_domain_count when no CorrelationTracker is wired.
        # Sorted for deterministic effective count (order-dependent algorithm).
        domain_list = sorted({sig.domain for sig in signals})
        if self._correlation_tracker:
            effective_count = self._compute_effective_domain_count(domain_list)
        else:
            effective_count = cross_domain_count  # backward-compatible fallback

        # Cross-domain diversity bonus (multiplicative, log-saturating)
        # 1 domain=0%, 2=~8%, 3=~13%, 4=~17%, saturates near 25%
        diversity_factor = 1.0 + 0.12 * math.log1p(max(0, effective_count - 1))
        boosted = geo_mean * diversity_factor

        return min(0.95, max(0.05, boosted))

    def _compute_effective_domain_count(self, domains: List[str]) -> float:
        """Compute effective number of independent domains, accounting for correlations.

        Strongly correlated domain pairs reduce the effective count — two domains
        that move together carry less independent information than two that don't.

        Algorithm:
        - First domain always counts fully (1.0).
        - Each additional domain adds credit based on its max |correlation| with
          all previously counted domains:
            |r| > 0.5  →  +0.5  (strongly correlated — half credit)
            |r| > 0.3  →  +0.7  (moderately correlated — partial credit)
            no data or |r| <= 0.3 →  +1.0  (independent — full credit)

        Correlation is looked up at the source level via CorrelationTracker and
        _DOMAIN_SOURCES, taking the maximum |correlation| across all source pairs
        for the two domains.

        Args:
            domains: List of domain names present in the convergence (e.g. ["macro", "insider"])

        Returns:
            Effective domain count as a float (always >= 1.0 if domains is non-empty)
        """
        if not domains:
            return 0.0
        if len(domains) == 1:
            return 1.0

        counted_domains: List[str] = [domains[0]]
        effective_count: float = 1.0

        for domain in domains[1:]:
            max_abs_corr = self._max_domain_correlation(domain, counted_domains)
            if max_abs_corr > STRONG_CORRELATION_THRESHOLD:
                effective_count += 0.5  # strongly correlated — half credit
            elif max_abs_corr > MODERATE_CORRELATION_THRESHOLD:
                effective_count += 0.7  # moderately correlated — partial credit
            else:
                effective_count += 1.0  # independent or no data — full credit
            counted_domains.append(domain)

        return effective_count

    def _max_domain_correlation(self, domain_a: str, other_domains: List[str]) -> float:
        """Return the maximum absolute correlation between domain_a and any domain in other_domains.

        Looks up correlations at source level using _DOMAIN_SOURCES and CorrelationTracker.
        Returns 0.0 if no correlation data is available (treated as independent).
        """
        sources_a = self._DOMAIN_SOURCES.get(domain_a, [])
        if not sources_a:
            return 0.0

        max_corr = 0.0
        for other_domain in other_domains:
            sources_b = self._DOMAIN_SOURCES.get(other_domain, [])
            if not sources_b:
                continue
            for src_a in sources_a:
                for src_b in sources_b:
                    corr = self._correlation_tracker.get_correlation(src_a, src_b)
                    if corr is not None:
                        max_corr = max(max_corr, abs(corr))

        return max_corr

    def _compute_coherence_score(self) -> dict:
        """Compute agreement ratio between directional signals across ALL domains.

        Used by Capability 3 (narrative coherence scoring) to detect when the
        market is sending contradictory signals — e.g. 5 bullish and 4 bearish
        domains simultaneously.  This does NOT look at signal strength; it is a
        simple directional vote count across all currently-held signals.

        Returns a dict with:
          coherence: float in [0.5, 1.0] — 1.0 = all agree, 0.5 = evenly split.
          dominant_direction: "bullish" or "bearish" (or "neutral" if no votes).
          bullish_count: number of domains with at least one bullish signal.
          bearish_count: number of domains with at least one bearish signal.
          contradiction_details: list of (domain, direction) for minority domains.
        """
        bullish_domains = []
        bearish_domains = []

        for domain, signals in self.signals.items():
            has_bullish = any(s.direction == "bullish" for s in signals)
            has_bearish = any(s.direction == "bearish" for s in signals)
            if has_bullish:
                bullish_domains.append(domain)
            if has_bearish:
                bearish_domains.append(domain)

        total = len(bullish_domains) + len(bearish_domains)
        if total == 0:
            return {
                "coherence": 1.0,
                "dominant_direction": "neutral",
                "bullish_count": 0,
                "bearish_count": 0,
                "contradiction_details": [],
            }

        bullish_count = len(bullish_domains)
        bearish_count = len(bearish_domains)

        # coherence = fraction of votes going to the majority side
        majority_count = max(bullish_count, bearish_count)
        coherence = majority_count / total  # range [0.5, 1.0]

        if bullish_count >= bearish_count:
            dominant = "bullish"
            minority_domains = [(d, "bearish") for d in bearish_domains]
        else:
            dominant = "bearish"
            minority_domains = [(d, "bullish") for d in bullish_domains]

        return {
            "coherence": coherence,
            "dominant_direction": dominant,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "contradiction_details": minority_domains,
        }

    def check_convergence(
        self,
        direction_filter: str = None
    ) -> List[ConvergenceAlert]:
        """
        Check for convergence and generate alerts.

        Args:
            direction_filter: Only check for specific direction (bullish/bearish)

        Returns:
            List of ConvergenceAlert objects
        """
        self._prune_old_signals()

        # Compute coherence once across all signals — shared by both directions.
        # Bearish signals dampen bullish confidence and vice versa.
        coherence_data = self._compute_coherence_score()

        # Publish CH_CONTRADICTION_DETECTED when signals materially disagree.
        # Completes Cap 3 (narrative coherence) — the sensor existed but never
        # published to the EventBus. Also feeds CausalReasoningEngine if wired.
        coherence = coherence_data.get("coherence", 1.0)
        if coherence < 0.7 and self._bus is not None:
            try:
                from mae_core.market.channels import CH_CONTRADICTION_DETECTED
                self._bus.publish(CH_CONTRADICTION_DETECTED, {
                    "coherence": coherence,
                    "dominant_direction": coherence_data.get("dominant_direction"),
                    "bullish_count": coherence_data.get("bullish_count", 0),
                    "bearish_count": coherence_data.get("bearish_count", 0),
                    "contradiction_details": coherence_data.get("contradiction_details", []),
                })
            except Exception:
                logger.debug("CH_CONTRADICTION_DETECTED publish failed", exc_info=True)

        # Feed domain pair correlations into CausalReasoningEngine.
        # This builds the causal graph that hypothesis_validator can later query.
        if self._causal_engine is not None:
            try:
                active_domains = [d for d, sigs in self.signals.items() if sigs]
                for i, domain_a in enumerate(active_domains):
                    for domain_b in active_domains[i + 1:]:
                        # Compute strength as product of max signal strengths
                        str_a = max((s.strength for s in self.signals[domain_a]), default=0)
                        str_b = max((s.strength for s in self.signals[domain_b]), default=0)
                        corr_strength = str_a * str_b
                        if corr_strength > 0.3:
                            self._causal_engine.observe_correlation(
                                domain_a, domain_b, corr_strength,
                                context={"source": "convergence_alerter"},
                            )
            except Exception:
                logger.debug("CausalReasoningEngine observation failed", exc_info=True)

        alerts = []

        # Check bullish convergence
        if direction_filter in (None, "bullish"):
            alert = self._check_direction_convergence("bullish", coherence_data)
            if alert:
                alerts.append(alert)

        # Check bearish convergence
        if direction_filter in (None, "bearish"):
            alert = self._check_direction_convergence("bearish", coherence_data)
            if alert:
                alerts.append(alert)

        # Alert deduplication — suppress re-alert within interval, per direction.
        # Using a dict keyed by direction so bullish and bearish each maintain
        # their own independent suppression window.  A threading.Lock guards the
        # dict so concurrent step-hook calls can't race each other.
        with self._alert_lock:
            now = datetime.now()
            filtered = []
            for alert in alerts:
                direction = alert.direction if hasattr(alert, "direction") else "neutral"
                last_time = self._last_alert_times.get(direction)
                if (last_time is not None
                        and (now - last_time).total_seconds() / 3600
                        < self._min_alert_interval_hours):
                    continue  # Suppress — same direction, too recent
                filtered.append(alert)
                self._last_alert_times[direction] = now
            alerts = filtered

        # Store alerts (capped to prevent unbounded growth)
        self.alerts.extend(alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-500:]

        # Log novel cross-domain discoveries
        for alert in alerts:
            if alert.cross_domain_count >= 3:
                self._log_discovery(alert)

        return alerts

    def _check_direction_convergence(
        self,
        direction: str,
        coherence_data: dict = None,
    ) -> Optional[ConvergenceAlert]:
        """Check for convergence in a specific direction.

        Directional signals (bullish/bearish) contribute domain presence, direction
        voting, strength, and confidence.

        Neutral signals (e.g. finra_short with short_ratio < threshold) contribute
        domain presence and confidence weighting ONLY — they do not cast a direction
        vote and are not counted toward avg_strength.  This ensures that a reliable
        volatility predictor like FINRA short-interest counts toward min_domains
        without polluting the directional signal.

        Capability 3 — Narrative coherence: if coherence_data is provided, the
        final confidence is multiplied by a coherence_multiplier derived from the
        global directional vote ratio.  Perfect agreement (coherence=1.0) leaves
        confidence unchanged; evenly split signals (coherence=0.5) halve it.
        Formula: coherence_multiplier = 0.5 + 0.5 * coherence → range [0.5, 1.0].
        """
        directional_signals = []   # Cast direction vote + strength + confidence
        neutral_signals = []       # Domain presence + confidence only
        domains_seen = set()
        categories_seen = set()
        effective_strengths = {}   # id(signal) -> effective strength (Cap 5 + 6)

        for domain, signals in self.signals.items():
            # Directional signals for the requested direction
            matching = [
                s for s in signals
                if s.direction == direction and s.strength >= self.min_strength
            ]

            # Neutral signals from this domain (direction-agnostic predictors)
            neutral = [
                s for s in signals
                if s.direction == "neutral" and s.strength >= self.min_strength
            ]

            if matching:
                # Cap 5: Select strongest by freshness-weighted strength.
                # A 2-hour-old signal at 0.8 beats a 70-hour-old signal at 0.85.
                strongest = max(
                    matching,
                    key=lambda s: s.strength * self._compute_freshness(s, domain)
                )
                max_eff = strongest.strength * self._compute_freshness(strongest, domain)

                # Cap 6: Intra-domain count boost — multiple confirming signals
                # from the same domain IS additional evidence.
                # 1 signal: 1.0x, 2: ~1.07x, 3: ~1.11x (log-saturating)
                count = len(matching)
                if count > 1:
                    max_eff *= (1 + 0.1 * math.log(count))

                directional_signals.append(strongest)
                effective_strengths[id(strongest)] = max_eff
                domains_seen.add(domain)
                categories_seen.add(self.domain_categories.get(domain, domain))
            elif neutral:
                # Cap 5: Select strongest neutral by freshness-weighted strength.
                strongest_neutral = max(
                    neutral,
                    key=lambda s: s.strength * self._compute_freshness(s, domain)
                )
                neutral_signals.append(strongest_neutral)
                domains_seen.add(domain)
                categories_seen.add(self.domain_categories.get(domain, domain))

        # Need minimum domains (directional + neutral combined)
        if len(domains_seen) < self.min_domains:
            # Emit partial convergence for ecosystem investigation
            if directional_signals and self._bus is not None:
                try:
                    self._bus.publish("market.intel.partial_convergence", {
                        "direction": direction,
                        "domains_seen": list(domains_seen),
                        "missing_domains": self._compute_missing_domains(domains_seen),
                        "signals": [{"source": s.source, "strength": s.strength,
                                     "symbol": getattr(s, "symbol", ""),
                                     "metadata": s.metadata} for s in directional_signals[:5]],
                        "min_domains_required": self.min_domains,
                    })
                except Exception:
                    pass  # Never block convergence check
            return None

        # Confidence is computed over ALL contributing signals so that
        # Thompson-weighted neutral sources (e.g. high-reliability FINRA short)
        # improve the estimate without inflating strength or direction.
        all_contributing = directional_signals + neutral_signals

        # Calculate cross-domain count (different categories)
        cross_domain_count = len(categories_seen)

        # Cap 5+6: Strength uses effective (freshness-weighted, count-boosted) values.
        # Neutral signals don't vote on strength — they only add domain presence.
        if directional_signals:
            avg_strength = sum(
                effective_strengths.get(id(s), s.strength)
                for s in directional_signals
            ) / len(directional_signals)
        else:
            # All domains are neutral — no directional conviction; don't alert
            return None

        final_confidence = self._compute_confidence(all_contributing, cross_domain_count)

        # Capability 3 — Apply coherence multiplier.
        # Bearish signals dampen bullish confidence and vice versa.
        # coherence_multiplier = 0.5 + 0.5 * coherence → range [0.5, 1.0].
        # Perfect agreement (coherence=1.0): unchanged.  Evenly split: halved.
        coherence = 1.0
        contradiction_details: list = []
        if coherence_data is not None:
            coherence = coherence_data.get("coherence", 1.0)
            contradiction_details = coherence_data.get("contradiction_details", [])
            coherence_multiplier = 0.5 + 0.5 * coherence
            final_confidence = max(0.05, min(0.95, final_confidence * coherence_multiplier))

        # Gift 3: Catalyst Calendar modifier — timing context affects confidence.
        # Extract primary ticker from signals' metadata for catalyst + cross-asset.
        primary_ticker = None
        primary_domain = None
        for sig in directional_signals:
            sym = sig.metadata.get("symbol", "")
            if sym:
                primary_ticker = sym
                primary_domain = sig.domain
                break

        if primary_ticker and self._catalyst_calendar is not None:
            try:
                modifier = self._catalyst_calendar.compute_catalyst_modifier(
                    primary_ticker, signal_domain=primary_domain or "",
                )
                final_confidence = max(0.05, min(0.95, final_confidence * modifier))
            except Exception:
                pass  # Graceful degradation

        # Gift 4: Cross-Asset Confirmation — correlated assets agree/disagree.
        # Score range: -1 to 1. Mapped to confidence multiplier: 0.4x to 1.2x.
        if primary_ticker and self._cross_asset_confirmer is not None:
            try:
                result = self._cross_asset_confirmer.check_confirmation(
                    primary_ticker, direction,
                )
                if result is not None:
                    score = getattr(result, "confirmation_score", 0.0)
                    cross_multiplier = 0.8 + 0.4 * score
                    final_confidence = max(0.05, min(0.95, final_confidence * cross_multiplier))
            except Exception:
                pass  # Graceful degradation

        # Gift 5: Deception detection — penalize confidence on suspected manipulation.
        if primary_ticker and self._deception_detector is not None:
            try:
                assessment = self._deception_detector.assess_signal_authenticity(primary_ticker)
                if assessment is not None and assessment.deception_probability > 0.5:
                    deception_penalty = 1.0 - assessment.deception_probability
                    final_confidence = max(0.05, final_confidence * deception_penalty)
            except Exception:
                pass  # Graceful degradation

        # Economic calendar suppression — reduce confidence during high-impact events.
        # FOMC, CPI, NFP create scheduled volatility that makes normal signals unreliable.
        # Multiplier: 0.5x during suppression windows (halves confidence).
        if self._economic_calendar is not None:
            try:
                if self._economic_calendar.is_in_suppression_window():
                    final_confidence = max(0.05, final_confidence * 0.5)
                    logger.info(
                        "Convergence: suppression window active — confidence reduced to %.2f",
                        final_confidence,
                    )
            except Exception:
                pass  # Graceful degradation

        # Gift 8: Pattern Archetype context — boost confidence when signals match a known archetype.
        if primary_ticker and self._pattern_archetype_engine is not None:
            try:
                matches = self._pattern_archetype_engine.scan_for_archetypes(
                    primary_ticker, signal_domains=list(domains_seen),
                )
                if matches:
                    best = max(matches, key=lambda m: m.match_score)
                    if best.match_score > 0.7:
                        final_confidence = min(0.95, final_confidence + 0.10)
            except Exception:
                pass  # Graceful degradation

        # Combo Thompson — historical domain combination reliability.
        # Different domain combos have vastly different win rates (8% to 67%).
        # The combo key tracks per-combination Thompson distributions.
        combo_key = "combo:" + "+".join(sorted(domains_seen))
        if self._thompson is not None:
            try:
                regime = self._get_regime()
                combo_dist = self._thompson.get_distribution(combo_key, regime)
                if combo_dist.samples >= 5:
                    combo_multiplier = 0.5 + combo_dist.mean
                    final_confidence = max(0.05, min(0.95, final_confidence * combo_multiplier))
            except Exception:
                pass  # Graceful degradation

        # Determine urgency based on velocity (directional signals only)
        avg_velocity = sum(abs(s.velocity) for s in directional_signals) / len(directional_signals)
        if avg_velocity > 0.1:
            urgency = "immediate"
        elif avg_velocity > 0.05:
            urgency = "hours"
        else:
            urgency = "days"

        # Temporal ordering — sort contributing signals by when they fired.
        # domain_sequence: domains ordered earliest-to-latest.
        # sequence_score: multiplier based on how well order matches known lags.
        try:
            domain_sequence = self._build_domain_sequence(all_contributing)
            sequence_score = self._compute_sequence_score(domain_sequence)
            # Apply sequence score to confidence (within existing [0.05, 0.95] bounds)
            if sequence_score != 1.0:
                final_confidence = max(0.05, min(0.95, final_confidence * sequence_score))
        except Exception:
            domain_sequence = []
            sequence_score = 1.0
            logger.debug("Temporal ordering failed gracefully", exc_info=True)

        # Quorum boost — collective agent agreement amplifies confidence.
        # Uses primary_ticker extracted earlier in this method.
        if primary_ticker:
            final_confidence = self._apply_quorum_boost(
                final_confidence, primary_ticker, direction
            )

        # Generate summary — note neutral signal count and coherence for transparency
        domain_list = ", ".join(sorted(domains_seen))
        neutral_note = f", +{len(neutral_signals)} neutral" if neutral_signals else ""
        coherence_note = f", coherence={coherence:.2f}" if coherence < 1.0 else ""
        seq_note = f", seq={sequence_score:.2f}" if sequence_score != 1.0 else ""
        summary = (
            f"{direction.upper()}: {len(domains_seen)} domains converging "
            f"({domain_list}) | {len(directional_signals)} directional"
            f"{neutral_note} | strength={avg_strength:.2f}, "
            f"confidence={final_confidence:.2f}, urgency={urgency}{coherence_note}{seq_note}"
        )

        # Create alert — signals list includes both directional and neutral
        # so callers can inspect the full contributing set
        self._alert_counter += 1
        alert = ConvergenceAlert(
            alert_id=f"CONV-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}",
            timestamp=datetime.now(),
            direction=direction,
            strength=avg_strength,
            confidence=final_confidence,
            domains_converging=sorted(domains_seen),
            signals=all_contributing,
            cross_domain_count=cross_domain_count,
            summary=summary,
            urgency=urgency,
            coherence=coherence,
            contradiction_details=contradiction_details,
            combo_key=combo_key,
            domain_sequence=domain_sequence,
            sequence_score=sequence_score,
        )

        return alert

    def get_domain_status(self) -> Dict[str, Dict]:
        """
        Get current status by domain.

        Returns:
            Dict mapping domain -> {direction, strength, signal_count}
        """
        self._prune_old_signals()

        status = {}
        for domain, signals in self.signals.items():
            if not signals:
                continue

            # Determine dominant direction
            bullish = [s for s in signals if s.direction == "bullish"]
            bearish = [s for s in signals if s.direction == "bearish"]

            if len(bullish) > len(bearish):
                dominant = "bullish"
                strength = sum(s.strength for s in bullish) / len(bullish) if bullish else 0
            elif len(bearish) > len(bullish):
                dominant = "bearish"
                strength = sum(s.strength for s in bearish) / len(bearish) if bearish else 0
            else:
                dominant = "neutral"
                strength = 0.5

            status[domain] = {
                "direction": dominant,
                "strength": round(strength, 3),
                "signal_count": len(signals),
                "bullish_count": len(bullish),
                "bearish_count": len(bearish),
                "category": self.domain_categories.get(domain, "unknown")
            }

        return status

    def _compute_missing_domains(self, domains_seen: set) -> list:
        """Return domains with current signals that haven't fired for this direction."""
        all_domains = set(self.signals.keys())
        return sorted(all_domains - domains_seen)

    def check_ticker_convergence_for(self, ticker: str) -> list:
        """Check convergence for a single ticker symbol."""
        return [
            a for a in self.check_ticker_convergence(min_domains=self.min_domains)
            if any(
                s.metadata.get("symbol") == ticker
                for s in getattr(a, "signals", [])
            )
        ]

    def check_ticker_convergence(self, min_domains: int = 2) -> List[ConvergenceAlert]:
        """Per-ticker convergence — more actionable than global domain convergence.

        Groups signals by symbol, then checks if multiple domains converge
        on the same ticker. A ticker with insider buying + hiring bullish +
        contract award across 3 domains is far stronger than 3 domains
        globally bullish on different stocks.

        Args:
            min_domains: Minimum different domains with signals on the same ticker

        Returns:
            List of ConvergenceAlert objects, one per ticker with sufficient convergence
        """
        self._prune_old_signals()

        # Group all signals by symbol
        by_ticker: Dict[str, Dict[str, List[Signal]]] = defaultdict(lambda: defaultdict(list))
        for domain, signals in self.signals.items():
            for sig in signals:
                symbol = sig.metadata.get("symbol", "")
                if symbol:
                    by_ticker[symbol][domain].append(sig)

        alerts = []
        for ticker, domain_signals in by_ticker.items():
            if len(domain_signals) < min_domains:
                continue

            # Check convergence for this ticker in each direction
            for direction in ("bullish", "bearish"):
                converging = []
                domains_seen = set()
                categories_seen = set()

                for domain, sigs in domain_signals.items():
                    matching = [s for s in sigs if s.direction == direction]
                    if matching:
                        strongest = max(matching, key=lambda s: s.strength)
                        converging.append(strongest)
                        domains_seen.add(domain)
                        categories_seen.add(self.domain_categories.get(domain, domain))

                if len(domains_seen) < min_domains:
                    continue

                avg_strength = sum(s.strength for s in converging) / len(converging)
                cross_domain_count = len(categories_seen)
                final_confidence = self._compute_confidence(converging, cross_domain_count)

                # Combo Thompson — same logic as global convergence path.
                combo_key = "combo:" + "+".join(sorted(domains_seen))
                if self._thompson is not None:
                    try:
                        regime = self._get_regime()
                        combo_dist = self._thompson.get_distribution(combo_key, regime)
                        if combo_dist.samples >= 5:
                            combo_multiplier = 0.5 + combo_dist.mean
                            final_confidence = max(0.05, min(0.95, final_confidence * combo_multiplier))
                    except Exception:
                        pass  # Graceful degradation

                # Temporal ordering — same logic as global convergence path.
                domain_sequence = []
                sequence_score = 1.0
                try:
                    domain_sequence = self._build_domain_sequence(converging)
                    sequence_score = self._compute_sequence_score(domain_sequence)
                    if sequence_score != 1.0:
                        final_confidence = max(0.05, min(0.95, final_confidence * sequence_score))
                except Exception:
                    pass  # Graceful degradation

                # Deception detection — same logic as global convergence path.
                if self._deception_detector is not None:
                    try:
                        deception_penalty = self._deception_detector.check(
                            ticker, direction, [s.source for s in converging]
                        )
                        if deception_penalty < 1.0:
                            final_confidence = max(0.05, final_confidence * deception_penalty)
                    except Exception:
                        pass  # Graceful degradation

                # Quorum boost — collective agent agreement amplifies confidence.
                # ticker is explicit in this per-ticker path.
                final_confidence = self._apply_quorum_boost(
                    final_confidence, ticker, direction
                )

                avg_velocity = sum(abs(s.velocity) for s in converging) / len(converging)
                if avg_velocity > 0.1:
                    urgency = "immediate"
                elif avg_velocity > 0.05:
                    urgency = "hours"
                else:
                    urgency = "days"

                domain_list = ", ".join(sorted(domains_seen))
                seq_note = f", seq={sequence_score:.2f}" if sequence_score != 1.0 else ""
                summary = (
                    f"TICKER {ticker} {direction.upper()}: {len(domains_seen)} domains "
                    f"({domain_list}) | strength={avg_strength:.2f}, "
                    f"confidence={final_confidence:.2f}, urgency={urgency}{seq_note}"
                )

                self._alert_counter += 1
                alert = ConvergenceAlert(
                    alert_id=f"TCKR-{ticker}-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}",
                    timestamp=datetime.now(),
                    direction=direction,
                    strength=avg_strength,
                    confidence=final_confidence,
                    domains_converging=sorted(domains_seen),
                    signals=converging,
                    cross_domain_count=cross_domain_count,
                    summary=summary,
                    urgency=urgency,
                    combo_key=combo_key,
                    domain_sequence=domain_sequence,
                    sequence_score=sequence_score,
                )
                alerts.append(alert)

        return alerts

    def get_convergence_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Get matrix of which domains agree with each other.

        Returns:
            Dict[domain_a][domain_b] = 1 (agree), -1 (disagree), 0 (no data)
        """
        status = self.get_domain_status()
        domains = list(status.keys())
        matrix = {}

        for domain_a in domains:
            matrix[domain_a] = {}
            dir_a = status[domain_a]["direction"]

            for domain_b in domains:
                if domain_a == domain_b:
                    matrix[domain_a][domain_b] = 1
                    continue

                dir_b = status[domain_b]["direction"]

                if dir_a == "neutral" or dir_b == "neutral":
                    matrix[domain_a][domain_b] = 0
                elif dir_a == dir_b:
                    matrix[domain_a][domain_b] = 1
                else:
                    matrix[domain_a][domain_b] = -1

        return matrix

    def get_actionable_summary(self) -> Dict:
        """
        Get actionable summary for trading decisions.

        Returns:
            Dict with direction recommendation, confidence, and reasoning
        """
        status = self.get_domain_status()

        bullish_domains = [d for d, s in status.items() if s["direction"] == "bullish"]
        bearish_domains = [d for d, s in status.items() if s["direction"] == "bearish"]

        bullish_strength = sum(status[d]["strength"] for d in bullish_domains) if bullish_domains else 0
        bearish_strength = sum(status[d]["strength"] for d in bearish_domains) if bearish_domains else 0

        # Count unique categories
        bullish_categories = set(status[d]["category"] for d in bullish_domains)
        bearish_categories = set(status[d]["category"] for d in bearish_domains)

        if len(bullish_domains) > len(bearish_domains) and len(bullish_categories) >= 2:
            recommendation = "bullish"
            avg_strength = bullish_strength / max(1, len(bullish_domains))
            n_cats = len(bullish_categories)
            diversity = 1.0 + 0.10 * math.log1p(n_cats)
            confidence = min(0.85, avg_strength * diversity)
            reasoning = f"{len(bullish_domains)} domains ({n_cats} categories) bullish"
        elif len(bearish_domains) > len(bullish_domains) and len(bearish_categories) >= 2:
            recommendation = "bearish"
            avg_strength = bearish_strength / max(1, len(bearish_domains))
            n_cats = len(bearish_categories)
            diversity = 1.0 + 0.10 * math.log1p(n_cats)
            confidence = min(0.85, avg_strength * diversity)
            reasoning = f"{len(bearish_domains)} domains ({n_cats} categories) bearish"
        else:
            recommendation = "neutral"
            confidence = 0.3
            reasoning = "Insufficient convergence - signals mixed or single-category"

        return {
            "recommendation": recommendation,
            "confidence": round(confidence, 3),
            "reasoning": reasoning,
            "bullish_domains": bullish_domains,
            "bearish_domains": bearish_domains,
            "total_signals": sum(s["signal_count"] for s in status.values())
        }

    def get_statistics(self) -> dict:
        """For HolonProxy.sense() delegation."""
        return {
            "domain_count": len(self.signals),
            "alert_count": len(self.alerts),
            "recent_alerts": [a.to_dict() for a in list(self.alerts)[-3:]],
        }

    def step(self) -> None:
        """Step hook for HolonProxy.act() delegation.

        Does not publish — bootstrap hook handles EventBus publishing
        with deduplication logic.
        """
        self.check_convergence()

    def to_dict(self) -> dict:
        """Export state for API/persistence."""
        return {
            "config": {
                "min_domains": self.min_domains,
                "min_strength": self.min_strength,
                "convergence_window_hours": self.convergence_window.total_seconds() / 3600
            },
            "domain_status": self.get_domain_status(),
            "actionable_summary": self.get_actionable_summary(),
            "recent_alerts": [a.to_dict() for a in self.alerts[-10:]]
        }

    def _log_discovery(self, alert: ConvergenceAlert) -> None:
        """Log a novel cross-domain convergence pattern to discovery_log.jsonl."""
        record = {
            "timestamp": alert.timestamp.isoformat(),
            "alert_id": alert.alert_id,
            "direction": alert.direction,
            "strength": round(alert.strength, 3),
            "confidence": round(alert.confidence, 3),
            "domains": alert.domains_converging,
            "cross_domain_count": alert.cross_domain_count,
            "signal_sources": [s.signal_id for s in alert.signals],
            "summary": alert.summary,
        }
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(_DISCOVERY_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug("Failed to write discovery log: %s", e)

    def save(self):
        """Persist state to disk."""
        if self.persistence_path:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)


def read_discoveries(
    max_entries: int = 100,
    min_strength: float = 0.0,
    direction: str = None,
) -> List[dict]:
    """
    Read recent discoveries from discovery_log.jsonl.

    Useful for:
    - Surfacing novel patterns to a dashboard
    - Seeding Thompson distributions with discovered signal combinations
    - Auditing what convergence patterns MIDGE has detected

    Args:
        max_entries: Maximum entries to return (most recent first)
        min_strength: Only return discoveries above this strength
        direction: Filter by "bullish" or "bearish" (None = all)

    Returns:
        List of discovery dicts, most recent first
    """
    if not _DISCOVERY_LOG.exists():
        return []

    entries = []
    try:
        with open(_DISCOVERY_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("strength", 0) < min_strength:
                        continue
                    if direction and entry.get("direction") != direction:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.debug("Failed to read discovery log: %s", e)

    # Most recent first, capped
    return entries[-max_entries:][::-1]


if __name__ == "__main__":
    from datetime import datetime, timedelta

    alerter = ConvergenceAlerter(min_domains=2, min_strength=0.5)

    # Simulate signals from multiple domains
    now = datetime.now()

    # Bullish signals from different domains
    alerter.record_signal("exec_buy_1", 0.8, "insider", "bullish", confidence=0.7)
    alerter.record_signal("whale_move_1", 0.75, "crypto", "bullish", confidence=0.65)
    alerter.record_signal("contract_award", 0.7, "government", "bullish", confidence=0.6)
    alerter.record_signal("reddit_hype", 0.6, "sentiment", "bullish", confidence=0.5)

    # One bearish signal
    alerter.record_signal("tech_indicator", 0.65, "technical", "bearish", confidence=0.55)

    print("=== Domain Status ===")
    for domain, status in alerter.get_domain_status().items():
        print(f"  {domain}: {status['direction']} (strength={status['strength']})")

    print("\n=== Convergence Check ===")
    alerts = alerter.check_convergence()
    for alert in alerts:
        print(f"  {alert.summary}")

    print("\n=== Actionable Summary ===")
    summary = alerter.get_actionable_summary()
    print(f"  Recommendation: {summary['recommendation']}")
    print(f"  Confidence: {summary['confidence']}")
    print(f"  Reasoning: {summary['reasoning']}")

    print("\n=== Convergence Matrix ===")
    matrix = alerter.get_convergence_matrix()
    domains = list(matrix.keys())
    print("       " + " ".join(f"{d[:6]:>7}" for d in domains))
    for d1 in domains:
        row = " ".join(f"{matrix[d1].get(d2, 0):>7}" for d2 in domains)
        print(f"{d1[:6]:>6} {row}")
