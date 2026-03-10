"""Trade processing functions for MIDGE market hooks.

Extracted from market_hooks.py — purely structural split.
Contains: _check_sweep_bypass, _write_paper_trade,
          _translate_and_log_executable_signal, _submit_to_alpaca.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")

# Sources eligible for the bypass path (backtest-validated, single-domain actionable).
# Min 1 domain instead of 3, quality gate replaces domain diversity requirement.
BYPASS_ELIGIBLE_SOURCES = {"session_sweep_ifvg"}


def _check_sweep_bypass(alerter, ctx: SimpleNamespace) -> None:
    """Direct output path for high-quality session sweep signals.

    Unlike standard convergence (min_domains=3, writes paper_trades.jsonl),
    this path uses min_domains=1 ticker convergence and applies a quality gate.
    Writes to data/midge/paper_trades_bypass.jsonl — a SEPARATE file.

    Gate: any contributing signal.source in BYPASS_ELIGIBLE_SOURCES
          AND alert quality >= 0.65 AND alert confidence >= 0.55.

    Dedup: ctx._bypass_dedup {"{direction}:{ticker}" -> datetime}, 4-hour window.
    """
    try:
        ticker_alerts = alerter.check_ticker_convergence(min_domains=1)
    except Exception:
        logger.debug("Sweep bypass ticker convergence failed", exc_info=True)
        return

    bypass_dedup = getattr(ctx, "_bypass_dedup", {})
    now = datetime.now()

    for alert in ticker_alerts:
        # Check that at least one contributing signal comes from an eligible source
        signals = getattr(alert, "signals", [])
        has_eligible = any(
            getattr(sig, "source", "") in BYPASS_ELIGIBLE_SOURCES
            for sig in signals
        )
        if not has_eligible:
            continue

        # Quality gate: pull from alert metadata or signals
        # Quality is stored in signal metadata from session_sweep_detector
        quality = 0.0
        for sig in signals:
            q = getattr(sig, "metadata", {}).get("quality", 0.0)
            if q > quality:
                quality = q

        confidence = getattr(alert, "confidence", 0.0)

        if quality < 0.65 or confidence < 0.55:
            logger.debug(
                "Sweep bypass rejected: quality=%.2f confidence=%.2f",
                quality, confidence,
            )
            continue

        # Resolve ticker and direction
        direction = getattr(alert, "direction", "neutral")
        if direction not in ("bullish", "bearish"):
            continue

        ticker = "UNKNOWN"
        for sig in signals:
            sym = getattr(sig, "metadata", {}).get("symbol", "")
            if sym:
                ticker = sym
                break

        # Dedup gate: same direction+ticker within 4 hours → skip
        dedup_key = f"{direction}:{ticker}"
        last_written = bypass_dedup.get(dedup_key)
        if last_written is not None and (now - last_written) < timedelta(hours=4):
            logger.debug("Bypass dedup suppressed: %s", dedup_key)
            continue

        # Write to separate bypass file
        try:
            alert_id = getattr(alert, "alert_id", None)
            signal_id = (
                f"BYP-{alert_id}" if alert_id
                else f"BYP-{now.strftime('%Y%m%d%H%M%S')}-{direction}"
            )
            domains = getattr(alert, "domains_converging", [])
            summary = getattr(alert, "summary", "")
            record = {
                "signal_id": signal_id,
                "asset": ticker,
                "asset_class": "futures",
                "direction": "buy" if direction == "bullish" else "sell",
                "confidence": round(float(confidence), 4),
                "quality": round(float(quality), 4),
                "bypass_reason": "backtest_validated",
                "contributing_signals": [
                    getattr(sig, "signal_id", "") for sig in signals
                    if getattr(sig, "signal_id", "")
                ],
                "domains": domains,
                "summary": summary,
                "generated_at": now.isoformat(),
            }
            bypass_path = Path("data/midge/paper_trades_bypass.jsonl")
            bypass_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bypass_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            bypass_dedup[dedup_key] = now
            ctx._bypass_dedup = bypass_dedup

            logger.info(
                "Sweep bypass trade written: %s %s quality=%.2f confidence=%.2f",
                direction.upper(), ticker, quality, confidence,
            )
        except Exception:
            logger.debug("Sweep bypass write failed", exc_info=True)


def _write_paper_trade(alert, ctx: SimpleNamespace) -> None:
    """Convert a high-confidence ConvergenceAlert into a TradeSignal and persist it.

    Called when alert passes confidence + strength + combo gates (see learning_config).

    Dedup gate: same direction+ticker combination is suppressed for 4 hours.
    Writes to data/midge/paper_trades.jsonl (atomic append).
    Optionally registers with OutcomeCollector to close the Thompson feedback loop.
    """
    try:
        from mae_core.market.signal import TradeSignal, MarketSignal

        # --- Resolve ticker from alert signals ---
        ticker = "MULTI"
        asset_class = "stock"
        for sig in getattr(alert, "signals", []):
            sym = getattr(sig, "metadata", {}).get("symbol", "")
            if sym:
                ticker = sym
                break
            # Session sweep signals carry asset_class=futures
            if getattr(sig, "source", "") in ("session_sweep", "session_sweep_ifvg"):
                asset_class = "futures"

        # Determine asset class from signal metadata if available
        for sig in getattr(alert, "signals", []):
            ac = getattr(sig, "metadata", {}).get("asset_class", "")
            if ac:
                asset_class = ac
                break

        # --- Dedup gate: same direction+ticker within 4h → skip ---
        dedup_key = f"{alert.direction}:{ticker}"
        dedup = getattr(ctx, "_paper_trade_dedup", {})
        now = datetime.now()
        last_written = dedup.get(dedup_key)
        if last_written is not None and (now - last_written) < timedelta(hours=4):
            logger.debug(
                "Paper trade dedup suppressed: %s (last: %s)",
                dedup_key, last_written.isoformat(timespec="seconds"),
            )
            return

        # --- Resolve direction (ConvergenceAlert uses bullish/bearish) ---
        raw_direction = getattr(alert, "direction", "neutral")
        if raw_direction == "bullish":
            trade_direction = "buy"
        elif raw_direction == "bearish":
            trade_direction = "sell"
        else:
            return  # Neutral alerts are not actionable

        # --- Resolve catalyst text ---
        summary = getattr(alert, "summary", None)
        domains_converging = getattr(alert, "domains_converging", [])
        if summary:
            catalyst = summary
        else:
            catalyst = (
                f"{raw_direction} convergence across "
                f"{len(domains_converging)} domains: {', '.join(domains_converging)}"
            )

        # --- Build contributing signal IDs ---
        contributing_signals = [
            getattr(sig, "signal_id", "") for sig in getattr(alert, "signals", [])
            if getattr(sig, "signal_id", "")
        ]

        # --- Generate signal_id ---
        alert_id = getattr(alert, "alert_id", None)
        if alert_id:
            signal_id = f"PT-{alert_id}"
        else:
            signal_id = f"PT-{now.strftime('%Y%m%d%H%M%S')}-{trade_direction}"

        # --- Kelly fraction (best-effort) ---
        kelly_fraction: float | None = None
        latest_kelly = getattr(ctx, "_latest_kelly", {}) or {}
        if isinstance(latest_kelly, dict) and latest_kelly.get("symbol") == ticker:
            kelly_fraction = latest_kelly.get("kelly_capped")

        # --- Instantiate TradeSignal ---
        trade_signal = TradeSignal(
            signal_id=signal_id,
            asset=ticker,
            asset_class=asset_class,
            direction=trade_direction,
            confidence=round(float(alert.confidence), 4),
            timeframe_days=5,
            catalyst=catalyst,
            contributing_signals=contributing_signals,
            hit_rate=0.0,
            generated_at=now,
        )

        # --- Serialize to JSONL (generated_at → ISO string) ---
        record = asdict(trade_signal)
        record["generated_at"] = trade_signal.generated_at.isoformat()
        if kelly_fraction is not None:
            record["kelly_fraction"] = round(float(kelly_fraction), 4)

        trade_path = Path("data/midge/paper_trades.jsonl")
        trade_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trade_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # --- Update dedup dict (evict entries older than 24h to prevent unbounded growth) ---
        dedup[dedup_key] = now
        _cutoff = now - timedelta(hours=24)
        ctx._paper_trade_dedup = {k: v for k, v in dedup.items() if v > _cutoff}

        logger.info(
            "Paper trade written: %s %s %s (confidence=%.2f, strength=%.2f, kelly=%s)",
            trade_direction.upper(), ticker, asset_class,
            alert.confidence, alert.strength,
            f"{kelly_fraction:.3f}" if kelly_fraction is not None else "n/a",
        )

        # --- Register with OutcomeCollector (closes Thompson feedback loop) ---
        outcome_collector = getattr(ctx, "_market_sensing_hook", None)
        if outcome_collector is not None:
            outcome_collector = getattr(outcome_collector, "_outcome_collector", None)
        if outcome_collector is None:
            # Fallback: check ctx directly (some setups store it there)
            outcome_collector = getattr(ctx, "outcome_collector", None)

        if outcome_collector is not None:
            try:
                # Synthesize a minimal MarketSignal for outcome tracking
                synthetic = MarketSignal(
                    signal_id=signal_id,
                    source="convergence_alert",
                    symbol=ticker,
                    asset_class=asset_class,
                    domain="convergence",
                    direction=raw_direction,
                    strength=float(alert.strength),
                    confidence=float(alert.confidence),
                    decay_rate=0.05,
                    timestamp=now,
                    received_at=now,
                    outcome_symbol=ticker,
                    outcome_window_days=trade_signal.timeframe_days,
                    metadata={"alert_id": getattr(alert, "alert_id", ""), "paper_trade": True},
                )
                outcome_collector.register_signals([synthetic])
                logger.debug("Paper trade %s registered with OutcomeCollector", signal_id)
            except Exception:
                logger.debug("OutcomeCollector registration for paper trade failed", exc_info=True)

        # --- Write plain-language alert for convergence-based paper trade ---
        try:
            from mae_core.market.plain_language import (
                format_convergence_alert, write_plain_alert,
            )
            _msg = format_convergence_alert(
                alert, ticker,
                window_days=trade_signal.timeframe_days,
            )
            write_plain_alert(
                _msg, ticker, raw_direction,
                source="convergence_alert",
                metadata={"confidence": float(alert.confidence),
                          "strength": float(alert.strength)},
            )
        except Exception:
            logger.debug("Plain-language convergence alert failed", exc_info=True)

    except Exception:
        logger.debug("_write_paper_trade failed", exc_info=True)


def _translate_and_log_executable_signal(alert, ctx: SimpleNamespace) -> None:
    """Translate a ConvergenceAlert into an ExecutableSignal and append it to
    data/midge/executable_signals.jsonl.

    This runs AFTER _write_paper_trade so it never blocks the existing paper-
    trade path.  All errors are swallowed -- failure here must never cascade.

    Pipeline:
      1. Resolve the ticker from alert.signals (same logic as _write_paper_trade).
      2. Fetch current price via ctx.price_fetcher (if available).
      3. Fetch 30 days of daily OHLCV from price_fetcher.get_daily_history().
      4. Compute 14-period ATR from those bars (compute_atr from ta_indicators).
      5. Call translate_alert() -> ExecutableSignal.
      6. Append JSON line to data/midge/executable_signals.jsonl.
    """
    try:
        from mae_core.market.execution.signal_translator import translate_alert
        from mae_core.market.edge.ta_indicators import compute_atr

        # --- Resolve ticker (mirrors _write_paper_trade logic) ---
        ticker = "MULTI"
        for sig in getattr(alert, "signals", []):
            sym = getattr(sig, "metadata", {}).get("symbol", "")
            if sym:
                ticker = sym
                break

        if ticker == "MULTI":
            logger.debug("_translate_and_log: no ticker resolved -- skipping")
            return

        # --- Current price ---
        price_fetcher = getattr(ctx, "price_fetcher", None)
        if price_fetcher is None:
            logger.debug("_translate_and_log: no price_fetcher on ctx -- skipping")
            return

        price_data = price_fetcher.get_current_price(ticker)
        if not price_data or not price_data.price:
            logger.debug("_translate_and_log: could not fetch price for %s", ticker)
            return

        current_price = float(price_data.price)

        # --- ATR from daily history ---
        history = price_fetcher.get_daily_history(ticker, days=30)
        atr = 0.0
        if len(history) >= 15:
            highs  = [p.high  for p in history]
            lows   = [p.low   for p in history]
            closes = [p.price for p in history]
            atr = compute_atr(highs, lows, closes, period=14)

        if atr <= 0:
            logger.debug("_translate_and_log: ATR unavailable for %s -- skipping", ticker)
            return

        # --- Translate ---
        alert_dict = alert.to_dict() if hasattr(alert, "to_dict") else {}
        # Ensure signals list present for ticker re-resolution inside translate_alert
        if not alert_dict.get("signals"):
            alert_dict["signals"] = [{"metadata": {"symbol": ticker}}]

        signal = translate_alert(alert_dict, current_price, atr)
        if signal is None:
            return

        # --- Persist ---
        out_path = Path("data/midge/executable_signals.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(signal.to_dict()) + "\n")

        logger.info(
            "Executable signal: %s %s entry=%.4f SL=%.4f TP=%.4f RR=%.2f pos=%.2f%%",
            signal.direction.upper(), signal.ticker,
            signal.entry_price, signal.stop_loss, signal.take_profit,
            signal.rr_ratio, signal.position_size_pct * 100,
        )

        # --- Submit to Alpaca (paper trading) ---
        _submit_to_alpaca(signal, ctx)

    except Exception:
        logger.debug("_translate_and_log_executable_signal failed", exc_info=True)


def _submit_to_alpaca(signal, ctx: SimpleNamespace) -> None:
    """Submit an ExecutableSignal to Alpaca paper trading.

    Guards:
    - Only fires if ctx.alpaca_client exists and is connected
    - Skips non-equity tickers (forex =X, futures =F, crypto -USD)
    - Skips if already holding a position in this ticker
    - All errors swallowed — failure here must never cascade
    """
    try:
        alpaca = getattr(ctx, "alpaca_client", None)
        if alpaca is None or not alpaca.connected:
            return

        ticker = signal.ticker

        # Only US equities — Alpaca doesn't trade forex/futures/crypto
        if any(suffix in ticker for suffix in ("=X", "=F", "-USD", ".X")):
            logger.debug("Alpaca: skipping non-equity ticker %s", ticker)
            return

        # Dedup — skip if we already have a position in this ticker
        existing = alpaca.get_positions()
        if any(p.symbol == ticker for p in existing):
            logger.debug("Alpaca: already holding %s — skipping", ticker)
            return

        # Position sizing — convert percentage to share count
        account = alpaca.get_account()
        if account is None:
            return

        dollar_amount = account.equity * signal.position_size_pct
        if dollar_amount < 1.0:
            logger.debug("Alpaca: position too small ($%.2f) for %s", dollar_amount, ticker)
            return

        qty = round(dollar_amount / signal.entry_price, 2)
        if qty <= 0:
            return

        side = "buy" if signal.direction == "long" else "sell"

        result = alpaca.submit_market_order(
            symbol=ticker,
            qty=qty,
            side=side,
            take_profit_price=round(signal.take_profit, 2),
            stop_loss_price=round(signal.stop_loss, 2),
            metadata={
                "source": "convergence_alert",
                "confidence": signal.confidence,
                "domains": signal.domains,
                "alert_id": signal.source_alert_id,
                "rr_ratio": signal.rr_ratio,
            },
        )

        if result:
            logger.info(
                "ALPACA PAPER TRADE: %s %s shares of %s @ ~%.2f | SL=%.2f TP=%.2f | order=%s",
                side.upper(), qty, ticker, signal.entry_price,
                signal.stop_loss, signal.take_profit, result.order_id,
            )
    except Exception:
        logger.debug("_submit_to_alpaca failed", exc_info=True)
