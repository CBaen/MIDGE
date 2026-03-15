"""Sensing reactive mixin — reactive convergence and fetch dispatch table.

Provides _trigger_reactive_convergence and _fetch_source.
Mixed into MarketSensingHook via SensingReactiveMixin.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("midge.market.sensing")


class SensingReactiveMixin:
    """Mixin providing reactive convergence checks and fetch source dispatch.

    Requires the host class to have:
      self._convergence_alerter, self._bus, self._cached_reactive_alerts,
      self._last_fetch_source, self._velocity_detector, self._filing_analyzer,
      self._form8k_sentiment, and all API client attributes
    """

    def _trigger_reactive_convergence(self, signals: list) -> None:
        """Fire convergence checks immediately after signals are ingested.

        Called from _collect_one() so that patterns are detected without
        waiting for the next step-tick. Never blocks signal collection.

        Three checks:
        1. Global check_convergence() — same path as step-tick
        2. Per-ticker check_ticker_convergence() for tickers in this batch
        Both publish to CH_CONVERGENCE and cache to _cached_reactive_alerts.
        """
        if self._convergence_alerter is None or not signals:
            return

        from mae_core.market.channels import CH_CONVERGENCE

        try:
            # 1. Global convergence
            alerts = self._convergence_alerter.check_convergence()
            for alert in alerts:
                alert_dict = alert.to_dict() if hasattr(alert, "to_dict") else {}
                if self._bus is not None:
                    try:
                        self._bus.publish(CH_CONVERGENCE, alert_dict)
                    except Exception:
                        logger.debug("Reactive convergence: failed to publish global alert", exc_info=True)
                self._cached_reactive_alerts.append(alert)
            if alerts:
                logger.info(
                    "Reactive convergence: %d global alert(s) after [%s] ingestion",
                    len(alerts), self._last_fetch_source,
                )
        except Exception:
            logger.debug("Reactive convergence: global check_convergence failed", exc_info=True)

        try:
            # 2. Per-ticker convergence for tickers that received signals
            tickers = {
                sig.metadata.get("symbol") or getattr(sig, "symbol", "")
                for sig in signals
            }
            tickers.discard("")
            if tickers:
                ticker_alerts = self._convergence_alerter.check_ticker_convergence()
                # Filter to only tickers from this batch
                relevant = [
                    a for a in ticker_alerts
                    if any(
                        s.metadata.get("symbol", "") in tickers
                        for s in getattr(a, "signals", [])
                    )
                ]
                for alert in relevant:
                    alert_dict = alert.to_dict() if hasattr(alert, "to_dict") else {}
                    if self._bus is not None:
                        try:
                            self._bus.publish(CH_CONVERGENCE, alert_dict)
                        except Exception:
                            logger.debug("Reactive convergence: failed to publish ticker alert", exc_info=True)
                    self._cached_reactive_alerts.append(alert)
                if relevant:
                    logger.info(
                        "Reactive convergence: %d ticker alert(s) for %s",
                        len(relevant), sorted(tickers),
                    )
        except Exception:
            logger.debug("Reactive convergence: ticker check_ticker_convergence failed", exc_info=True)

    def _fetch_source(self, source_name: str) -> list:
        """Fetch from a single source. Runs in ThreadPoolExecutor.

        Returns list of MarketSignal objects.
        """
        from mae_core.market.signal import (
            from_insider_trade,
            from_form8k_event,
            from_congressional_trade,
            from_senate_trade,
            from_hiring_signal,
            from_government_contract,
            from_contract_opportunity,
            from_cluster_signal,
            from_correlation_signal,
            from_contract_prediction,
            from_social_sentiment,
            from_short_interest,
            from_filing_keyword,
            from_news_sentiment,
            from_earnings_event,
            from_macro_indicator,
            from_session_sweep,
            from_ta_signal,
            from_cot_positioning,
            from_stocktwits_sentiment,
            from_vix_structure,
            from_trends_signal,
            from_economic_event,
            from_analyst_recommendation,
            from_order_flow,
            from_fractal_resonance,
            from_crypto_signal,
            from_openinsider,
            from_13f_holding,
            from_activist_filing,
            from_finviz_unusual_volume,
            from_finviz_short_squeeze,
            from_suppression_event,
            from_economic_surprise,
        )
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_insider
        from mae_core.market.signal_adapters.layer6 import from_social_text_signal
        from mae_core.market.sensing_fetchers import (
            fetch_sec_form4, fetch_sec_form8k, fetch_congressional, fetch_senate,
            fetch_hiring, fetch_usa_spending, fetch_sam_gov, fetch_social_sentiment,
            fetch_finra_short, fetch_sec_efts, fetch_finnhub, fetch_fred,
            fetch_session_sweep, fetch_ta_indicators, fetch_cot, fetch_stocktwits,
            fetch_vix, fetch_trends, fetch_finnhub_extras, fetch_order_flow,
            fetch_fractal_resonance, fetch_crypto_prices, fetch_crypto_exchange,
            fetch_openinsider, fetch_13f_holdings, fetch_finviz, fetch_economic_calendar,
            fetch_eia, fetch_congress_legislation, fetch_massive_snapshot,
            fetch_social_text, fetch_yahoo_rss, fetch_usda, fetch_fred_yields,
            fetch_binance_funding,
            fetch_kalshi_movers,
            fetch_cboe_options,
            fetch_crypto_fear_greed,
        )
        from mae_core.market.sensing_lifecycle import enrich_signal

        signals = []

        if source_name == "sec_form4":
            signals = fetch_sec_form4(
                self._sec_client, self._watchlist, from_insider_trade,
                cluster_detector=self._cluster_detector,
                cluster_converter=from_cluster_signal,
            )

        elif source_name == "sec_form8k":
            signals = fetch_sec_form8k(self._sec_client, self._watchlist, from_form8k_event)

        elif source_name == "congressional":
            signals = fetch_congressional(
                self._congress_client, from_congressional_trade,
                politician_tracker=self._politician_tracker,
                correlation_converter=from_correlation_signal,
            )

        elif source_name == "senate":
            signals = fetch_senate(self._senate_client, from_senate_trade)

        elif source_name == "hiring":
            signals = fetch_hiring(
                self._job_tracker, self._watchlist, from_hiring_signal,
                contract_predictor=self._contract_predictor,
                prediction_converter=from_contract_prediction,
            )

        elif source_name == "usa_spending":
            signals = fetch_usa_spending(self._usa_spending, self._watchlist, from_government_contract)

        elif source_name == "sam_gov_and_prices":
            signals = fetch_sam_gov(self._sam_gov, self._watchlist, from_contract_opportunity)

        elif source_name == "social_sentiment":
            signals = fetch_social_sentiment(self._apewisdom, self._watchlist, from_social_sentiment)

        elif source_name == "finra_short":
            signals = fetch_finra_short(self._finra_client, self._watchlist, from_short_interest)

        elif source_name == "sec_efts":
            signals = fetch_sec_efts(self._sec_efts, from_filing_keyword)

        elif source_name == "finnhub":
            signals = fetch_finnhub(self._finnhub, self._watchlist, from_news_sentiment, from_earnings_event)

        elif source_name == "fred_macro":
            signals = fetch_fred(self._fred, from_macro_indicator)

        elif source_name == "session_sweep":
            signals = fetch_session_sweep(self._session_sweep_detector, from_session_sweep)

        elif source_name == "ta_indicators":
            signals = fetch_ta_indicators(self._ta_indicators, self._price_fetcher, self._watchlist, from_ta_signal)

        elif source_name == "cot_positioning":
            signals = fetch_cot(self._cot_client, self._watchlist, from_cot_positioning)

        elif source_name == "stocktwits":
            signals = fetch_stocktwits(self._stocktwits_client, self._watchlist, from_stocktwits_sentiment)

        elif source_name == "vix_structure":
            signals = fetch_vix(self._vix_client, from_vix_structure)

        elif source_name == "google_trends":
            signals = fetch_trends(self._trends_client, self._watchlist, from_trends_signal)

        elif source_name == "finnhub_extras":
            signals = fetch_finnhub_extras(
                self._finnhub, self._watchlist, from_economic_event, from_analyst_recommendation,
                surprise_converter=from_economic_surprise,
            )

        elif source_name == "order_flow":
            signals = fetch_order_flow(self._order_flow_detector, self._watchlist, from_order_flow)

        elif source_name == "fractal_resonance":
            signals = fetch_fractal_resonance(
                self._fractal_resonance_detector, self._watchlist, from_fractal_resonance
            )

        elif source_name == "crypto_prices":
            signals = fetch_crypto_prices(self._coingecko_client, from_crypto_signal)

        elif source_name == "crypto_exchange":
            signals = fetch_crypto_exchange(self._coincap_client, from_crypto_signal)

        elif source_name == "binance_funding":
            from mae_core.market.signal_adapters.wave2_3_technical import from_binance_funding
            signals = fetch_binance_funding(self._binance_funding_client, from_binance_funding)

        elif source_name == "kalshi_market":
            from mae_core.market.signal_adapters.wave2_3_technical import from_kalshi_mover
            signals = fetch_kalshi_movers(self._kalshi_client, from_kalshi_mover)

        elif source_name == "openinsider":
            signals = fetch_openinsider(self._openinsider_client, from_openinsider)

        elif source_name == "institutional_13f":
            from mae_core.market.signal_adapters.wave2_3_insider import from_13f_filer_activity
            signals = fetch_13f_holdings(self._edgar_enhanced_client, from_13f_holding, from_activist_filing, from_13f_filer_activity)

        elif source_name == "finviz":
            signals = fetch_finviz(
                self._finviz_client,
                from_finviz_unusual_volume,
                from_finviz_short_squeeze,
                from_finviz_insider,
            )

        elif source_name == "economic_calendar":
            signals = fetch_economic_calendar(self._economic_calendar_client, from_suppression_event)

        elif source_name == "massive_snapshot":
            from mae_core.market.signal_adapters.wave2_3 import from_massive_snapshot
            signals = fetch_massive_snapshot(self._massive_client, self._watchlist, from_massive_snapshot)

        elif source_name == "eia_energy":
            from mae_core.market.signal_adapters.market_data import from_energy_indicator
            signals = fetch_eia(self._eia_client, from_energy_indicator)

        elif source_name == "congress_legislation":
            from mae_core.market.signal_adapters.market_data import from_legislative_indicator
            signals = fetch_congress_legislation(self._congress_gov_client, from_legislative_indicator)

        elif source_name == "social_text":
            signals = fetch_social_text(
                self._social_text_analyzer, self._watchlist, from_social_text_signal
            )

        elif source_name == "yahoo_rss":
            from mae_core.market.signal_adapters.layer6 import from_yahoo_rss_signal
            signals = fetch_yahoo_rss(self._yahoo_rss_client, self._watchlist, from_yahoo_rss_signal)

        elif source_name == "usda_agriculture":
            from mae_core.market.signal_adapters.market_data import from_usda_indicator
            signals = fetch_usda(self._usda_client, from_usda_indicator)

        elif source_name == "fred_yields":
            from mae_core.market.signal_adapters.market_data import from_fred_yield
            signals = fetch_fred_yields(self._fred, from_fred_yield)

        elif source_name == "cboe_options":
            signals = fetch_cboe_options(getattr(self, "_cboe_options_client", None))

        elif source_name == "crypto_fear_greed":
            signals = fetch_crypto_fear_greed(getattr(self, "_crypto_fear_greed_client", None))

        # Enrich in background thread (velocity, filing-time, Ollama sentiment)
        # Moved from _collect_results() so Ollama's 15s timeout doesn't block
        # the main step loop. Thread-safe: only mutates signal objects.
        for sig in signals:
            enrich_signal(
                sig,
                self._velocity_detector,
                self._filing_analyzer,
                self._form8k_sentiment,
                self._thompson_sampler,
            )

        return signals
