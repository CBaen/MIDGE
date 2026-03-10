"""Historical event search for HypothesisValidator.

Contains:
- _find_trigger_events: scan signal archive for source_a events
- _find_composite_trigger_events: scan for co-firing source_a + conjunct_source
- _check_event_outcome: look up outcome matching a trigger event
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def find_trigger_events(
    hypothesis,
    signals_dir: Path,
    lookback_days: int,
) -> List[dict]:
    """Find historical instances where source_a fired.

    Scans signal archive JSONL files for signals matching
    hypothesis.trigger.source_a within lookback window.
    """
    cutoff = datetime.now() - timedelta(days=lookback_days)
    events = []

    if not signals_dir.exists():
        return events

    for jsonl_file in sorted(signals_dir.glob("*.jsonl")):
        try:
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    source = record.get("source", "")
                    if source != hypothesis.trigger.source_a:
                        continue

                    ts_str = record.get("timestamp", "")
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except ValueError:
                        continue

                    if ts < cutoff:
                        continue

                    events.append(record)
        except Exception:
            continue

    return events


def find_composite_trigger_events(
    hypothesis,
    signals_dir: Path,
    lookback_days: int,
) -> List[dict]:
    """Find historical instances where BOTH source_a AND conjunct_source fired.

    For each source_a event, checks whether conjunct_source also fired
    within +/- lag_days/2 of the source_a event. Only events where both
    conditions are met are returned.

    This ensures composite hypotheses are validated against co-firing
    patterns, not just single-source events.
    """
    cutoff = datetime.now() - timedelta(days=lookback_days)

    # First pass: collect all source_a events
    source_a_events = find_trigger_events(hypothesis, signals_dir, lookback_days)
    if not source_a_events:
        return []

    # Second pass: collect all conjunct_source events for window matching
    conjunct_source = hypothesis.trigger.conjunct_source
    lag = hypothesis.trigger.lag_days
    half_window = timedelta(days=max(1, lag // 2))

    conjunct_events: List[dict] = []
    if signals_dir.exists():
        for jsonl_file in sorted(signals_dir.glob("*.jsonl")):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if record.get("source", "") != conjunct_source:
                            continue

                        ts_str = record.get("timestamp", "")
                        if not ts_str:
                            continue
                        try:
                            ts = datetime.fromisoformat(ts_str)
                        except ValueError:
                            continue

                        if ts < cutoff:
                            continue

                        conjunct_events.append(record)
            except Exception:
                continue

    if not conjunct_events:
        return []

    # Third pass: keep only source_a events that have a matching conjunct event
    matched_events = []
    for event_a in source_a_events:
        ts_a_str = event_a.get("timestamp", "")
        if not ts_a_str:
            continue
        try:
            ts_a = datetime.fromisoformat(ts_a_str)
        except ValueError:
            continue

        symbol_a = event_a.get("symbol", "")
        window_start = ts_a - half_window
        window_end = ts_a + half_window

        for event_c in conjunct_events:
            # Symbol must match if source_a had a symbol
            if symbol_a and event_c.get("symbol", "") != symbol_a:
                continue

            ts_c_str = event_c.get("timestamp", "")
            if not ts_c_str:
                continue
            try:
                ts_c = datetime.fromisoformat(ts_c_str)
            except ValueError:
                continue

            if window_start <= ts_c <= window_end:
                matched_events.append(event_a)
                break  # One conjunct match per source_a event is enough

    return matched_events


def check_event_outcome(
    trigger_event: dict,
    hypothesis,
    outcomes_path: Path,
) -> Optional[tuple]:
    """Check if a trigger event resulted in a successful prediction.

    Looks for outcome records in outcomes.jsonl where the source matches
    hypothesis.trigger.source_b and the timing aligns with the lag window.

    Returns (pct_return, success) or None if no matching outcome found.
    """
    trigger_ts_str = trigger_event.get("timestamp", "")
    if not trigger_ts_str:
        return None
    try:
        trigger_ts = datetime.fromisoformat(trigger_ts_str)
    except ValueError:
        return None

    trigger_symbol = trigger_event.get("symbol", "")
    lag = hypothesis.trigger.lag_days
    expected_outcome_start = trigger_ts + timedelta(days=max(1, lag - 10))
    expected_outcome_end = trigger_ts + timedelta(days=lag + 10)

    if not outcomes_path.exists():
        return None

    try:
        with open(outcomes_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    outcome = json.loads(line)
                except json.JSONDecodeError:
                    continue

                outcome_source = outcome.get("source", "")
                if outcome_source != hypothesis.trigger.source_b:
                    continue

                # Symbol match (if trigger had a symbol)
                if trigger_symbol and outcome.get("symbol", "") != trigger_symbol:
                    continue

                predicted_at_str = outcome.get("predicted_at", "")
                if not predicted_at_str:
                    continue
                try:
                    predicted_at = datetime.fromisoformat(predicted_at_str)
                except ValueError:
                    continue

                if expected_outcome_start <= predicted_at <= expected_outcome_end:
                    pct = outcome.get("price_change_pct", 0.0)
                    success = outcome.get("success", False)

                    # Adjust for hypothesis direction
                    if hypothesis.trigger.direction == "negative":
                        # Negative correlation: source_a up → source_b down
                        direction_match = pct < 0
                    elif hypothesis.trigger.direction == "positive":
                        direction_match = pct > 0
                    else:
                        direction_match = True

                    return (pct, success and direction_match)
    except Exception:
        pass

    return None
