"""Active Tracker — continuous monitoring of predicted assets.

When MIDGE flags an asset, she doesn't walk away for 14 days.
She watches it: tracking price, detecting confirmation/failure,
discovering new patterns during the watch, and learning from outcomes.

Status transitions:
  tracking -> confirming -> confirmed | failed | expired

Rate limiting: 5-minute minimum between price fetches per asset.
Capacity: max 20 tracked assets (oldest expired/confirmed evicted first).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_TRACKED = 20
PRICE_COOLDOWN_SECONDS = 300  # 5 minutes between fetches per asset
CONFIRM_THRESHOLD_FRACTION = 0.5  # 50% of expected magnitude = confirming
FAIL_THRESHOLD_PCT = 3.0  # 3% wrong direction = failed


@dataclass
class TrackedAsset:
    """An asset being actively monitored after a pattern detection."""
    symbol: str
    direction: str  # "bullish" or "bearish"
    entry_price: float
    expected_window_days: int
    template_ids: list[str] = field(default_factory=list)
    expected_magnitude_pct: float = 5.0  # expected % move
    status: str = "tracking"  # tracking | confirming | confirmed | failed | expired
    created_at: str = ""
    updated_at: str = ""
    mfe_pct: float = 0.0  # max favorable excursion
    mae_pct: float = 0.0  # max adverse excursion
    current_price: float = 0.0
    current_pct: float = 0.0
    last_price_check: str = ""
    confirmation_signals: list[dict] = field(default_factory=list)
    tier: str = ""
    stack_confidence: float = 0.0
    n_patterns: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrackedAsset:
        # Filter to known fields only
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


class ActiveTracker:
    """Manages a registry of actively monitored assets.

    Wired into the step hook cadence — called every N steps to
    check prices, update statuses, and discover new patterns.
    """

    def __init__(
        self,
        price_fetcher=None,
        outcome_collector=None,
        pattern_library=None,
        data_dir: Optional[Path] = None,
    ):
        self._price_fetcher = price_fetcher
        self._outcome_collector = outcome_collector
        self._pattern_library = pattern_library
        self._data_dir = data_dir or Path("data/midge")
        self._path = self._data_dir / "active_tracking.json"
        self._tracked: dict[str, TrackedAsset] = {}  # dedup_key -> TrackedAsset
        self._load()

    @property
    def count(self) -> int:
        return len(self._tracked)

    @property
    def active_symbols(self) -> list[str]:
        return [t.symbol for t in self._tracked.values()
                if t.status in ("tracking", "confirming")]

    def register(self, stack, entry_price: float) -> Optional[TrackedAsset]:
        """Register a pattern stack for active tracking.

        Returns TrackedAsset if registered, None if duplicate or at capacity.
        """
        if entry_price <= 0:
            return None

        dedup_key = f"{stack.symbol}:{stack.direction}"
        if dedup_key in self._tracked:
            existing = self._tracked[dedup_key]
            if existing.status in ("tracking", "confirming"):
                return None  # Already tracking this symbol+direction

        # Evict if at capacity
        if len(self._tracked) >= MAX_TRACKED:
            self._evict_one()
            if len(self._tracked) >= MAX_TRACKED:
                return None  # Still full after eviction attempt

        # Compute expected window from templates
        windows = []
        template_ids = []
        for act in stack.activations:
            tmpl = act.template
            template_ids.append(getattr(tmpl, "template_id", ""))
            w = getattr(tmpl, "expected_move_window_days", None)
            if w is not None:
                windows.append(w)
        if windows:
            windows.sort()
            window = windows[len(windows) // 2]
        else:
            window = 14

        # Expected magnitude from template avg_move_pct
        magnitudes = [getattr(act.template, "avg_move_pct", 5.0)
                      for act in stack.activations]
        avg_mag = sum(magnitudes) / len(magnitudes) if magnitudes else 5.0

        asset = TrackedAsset(
            symbol=stack.symbol,
            direction=stack.direction,
            entry_price=entry_price,
            expected_window_days=window,
            template_ids=template_ids,
            expected_magnitude_pct=max(3.0, avg_mag),
            current_price=entry_price,
            tier=getattr(stack, "tier", ""),
            stack_confidence=getattr(stack, "stack_confidence", 0.0),
            n_patterns=len(stack.activations),
        )
        self._tracked[dedup_key] = asset
        self._save()
        logger.info(
            "Active tracking started: %s %s @ $%.2f (window=%dd, expected=%.1f%%)",
            stack.symbol, stack.direction, entry_price, window, avg_mag,
        )
        return asset

    def check_prices(self) -> list[dict]:
        """Check prices for all actively tracked assets.

        Returns list of status change events (for logging/alerting).
        Rate limited: skips assets checked within PRICE_COOLDOWN_SECONDS.
        """
        if not self._price_fetcher:
            return []

        events = []
        now = datetime.now()

        # Collect symbols needing price checks
        to_check = {}
        for key, asset in self._tracked.items():
            if asset.status not in ("tracking", "confirming"):
                continue
            if asset.last_price_check:
                try:
                    last = datetime.fromisoformat(asset.last_price_check)
                    if (now - last).total_seconds() < PRICE_COOLDOWN_SECONDS:
                        continue
                except (ValueError, TypeError):
                    pass
            to_check[key] = asset

        if not to_check:
            return events

        # Batch price fetch
        symbols = list(set(a.symbol for a in to_check.values()))
        try:
            prices = {}
            for sym in symbols:
                try:
                    data = self._price_fetcher.get_current_price(sym)
                    if data and data.price > 0:
                        prices[sym] = data.price
                except Exception:
                    logger.debug("Price fetch failed for %s", sym, exc_info=True)
        except Exception:
            logger.debug("Batch price fetch failed", exc_info=True)
            return events

        # Update each tracked asset
        for key, asset in to_check.items():
            price = prices.get(asset.symbol)
            if price is None:
                continue

            old_status = asset.status
            asset.current_price = price
            asset.last_price_check = now.isoformat()

            # Calculate % change from entry
            pct = ((price - asset.entry_price) / asset.entry_price) * 100
            if asset.direction == "bearish":
                pct = -pct  # Favorable = price going down for bearish
            asset.current_pct = round(pct, 2)

            # Update MFE/MAE
            if pct > asset.mfe_pct:
                asset.mfe_pct = round(pct, 2)
            if pct < asset.mae_pct:
                asset.mae_pct = round(pct, 2)

            # Status transitions
            new_status = self._compute_status(asset, pct)
            if new_status != old_status:
                asset.status = new_status
                asset.updated_at = now.isoformat()
                event = {
                    "symbol": asset.symbol,
                    "direction": asset.direction,
                    "old_status": old_status,
                    "new_status": new_status,
                    "current_pct": asset.current_pct,
                    "price": price,
                    "entry_price": asset.entry_price,
                }
                events.append(event)
                logger.info(
                    "Tracking status change: %s %s -> %s (%.1f%% move)",
                    asset.symbol, old_status, new_status, pct,
                )

                # Force-grade on terminal status
                if new_status in ("confirmed", "failed"):
                    self._force_grade(asset, pct)

        # Check for expired assets
        for key, asset in self._tracked.items():
            if asset.status not in ("tracking", "confirming"):
                continue
            try:
                created = datetime.fromisoformat(asset.created_at)
                if (now - created).days >= asset.expected_window_days:
                    old_status = asset.status
                    asset.status = "expired"
                    asset.updated_at = now.isoformat()
                    events.append({
                        "symbol": asset.symbol,
                        "direction": asset.direction,
                        "old_status": old_status,
                        "new_status": "expired",
                        "current_pct": asset.current_pct,
                    })
                    logger.info(
                        "Tracking expired: %s after %d days (%.1f%% move)",
                        asset.symbol, asset.expected_window_days, asset.current_pct,
                    )
                    self._force_grade(asset, asset.current_pct)
            except (ValueError, TypeError):
                pass

        if events:
            self._save()

        return events

    def _compute_status(self, asset: TrackedAsset, favorable_pct: float) -> str:
        """Compute status based on price movement."""
        confirm_threshold = asset.expected_magnitude_pct
        partial_threshold = confirm_threshold * CONFIRM_THRESHOLD_FRACTION

        if favorable_pct >= confirm_threshold or favorable_pct >= 5.0:
            return "confirmed"
        if favorable_pct <= -FAIL_THRESHOLD_PCT:
            return "failed"
        if favorable_pct >= partial_threshold and asset.status == "tracking":
            return "confirming"
        return asset.status  # no change

    def _force_grade(self, asset: TrackedAsset, pct: float) -> None:
        """Force-grade the prediction via OutcomeCollector.

        Records an immediate outcome by writing a prediction with window_days=0
        (already matured) so the next evaluate() call grades it. Uses the public
        OutcomeTracker.record_prediction() + Thompson update path directly.
        """
        if not self._outcome_collector:
            return
        try:
            success = pct >= 5.0 or (asset.status == "confirmed")
            oc = self._outcome_collector
            # Use OutcomeTracker directly — it's always at oc.tracker
            tracker = getattr(oc, "tracker", None)
            if tracker is None:
                return

            # Determine source key for Thompson routing
            source = "pattern_stack:" + "+".join(sorted(asset.template_ids[:3])) if asset.template_ids else "active_tracker"
            direction = "up" if asset.direction == "bullish" else "down"

            # Update Thompson Sampler directly (most reliable path)
            ts = getattr(tracker, "thompson_sampler", None)
            if ts is not None:
                try:
                    regime = "default"
                    rc = getattr(tracker, "regime_classifier", None)
                    if rc is not None:
                        try:
                            regime = rc.classify()
                        except Exception:
                            pass
                    ts.update(source, success=success, regime=regime)
                except Exception:
                    logger.debug("Force-grade Thompson update failed for %s", asset.symbol, exc_info=True)

            # Append to outcomes.jsonl so the result is recorded
            outcome = {
                "signal_id": f"force_grade:{asset.symbol}:{asset.created_at}",
                "source": source,
                "symbol": asset.symbol,
                "direction": direction,
                "predicted_at": asset.created_at,
                "evaluated_at": datetime.now().isoformat(),
                "window_days": asset.expected_window_days,
                "price_change_pct": round(pct, 4),
                "success": success,
                "min_move_threshold": 5.0,
                "force_graded": True,
            }
            tracker._append_outcome(outcome)

            # Fire on_outcome callback (closes pattern_library feedback loop)
            if tracker.on_outcome is not None:
                try:
                    fake_pred = {
                        "source": source,
                        "symbol": asset.symbol,
                        "metadata": {"template_ids": asset.template_ids},
                    }
                    tracker.on_outcome(fake_pred, success, pct)
                except Exception:
                    logger.debug("on_outcome callback failed in force_grade", exc_info=True)

            logger.info(
                "Force-graded %s: %s (%.1f%%)",
                asset.symbol, "WIN" if success else "LOSS", pct,
            )
        except Exception:
            logger.debug("Force grade failed for %s", asset.symbol, exc_info=True)

    def _evict_one(self) -> None:
        """Evict one tracked asset to make room. Prefers terminal statuses."""
        # First try terminal statuses (confirmed/failed/expired)
        for key in list(self._tracked):
            if self._tracked[key].status in ("confirmed", "failed", "expired"):
                del self._tracked[key]
                return
        # Then evict oldest tracking
        oldest_key = None
        oldest_time = None
        for key, asset in self._tracked.items():
            try:
                created = datetime.fromisoformat(asset.created_at)
                if oldest_time is None or created < oldest_time:
                    oldest_time = created
                    oldest_key = key
            except (ValueError, TypeError):
                oldest_key = key
                break
        if oldest_key:
            del self._tracked[oldest_key]

    def get_status_summary(self) -> list[dict]:
        """Return summary of all tracked assets for reporting."""
        return [
            {
                "symbol": a.symbol,
                "direction": a.direction,
                "status": a.status,
                "current_pct": a.current_pct,
                "mfe_pct": a.mfe_pct,
                "mae_pct": a.mae_pct,
                "entry_price": a.entry_price,
                "current_price": a.current_price,
                "window_days": a.expected_window_days,
                "tier": a.tier,
                "n_patterns": a.n_patterns,
            }
            for a in self._tracked.values()
        ]

    def get_statistics(self) -> dict:
        """Summary statistics."""
        statuses = {}
        for a in self._tracked.values():
            statuses[a.status] = statuses.get(a.status, 0) + 1
        return {
            "total_tracked": len(self._tracked),
            "by_status": statuses,
            "active_symbols": self.active_symbols,
        }

    # ── Persistence ────────────────────────────────────────────────

    def _load(self) -> None:
        """Load tracked assets from disk."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                for key, d in data.items():
                    self._tracked[key] = TrackedAsset.from_dict(d)
                if self._tracked:
                    logger.info("Restored %d tracked assets", len(self._tracked))
            except Exception:
                logger.debug("Failed to load active tracking", exc_info=True)

    def _save(self) -> None:
        """Persist tracked assets to disk."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            data = {key: asset.to_dict() for key, asset in self._tracked.items()}
            self._path.write_text(json.dumps(data, indent=2))
        except Exception:
            logger.debug("Failed to save active tracking", exc_info=True)
