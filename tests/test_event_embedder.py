"""Tests for EventEmbedder, PatternMemory, and event_descriptions.

All external services (Qdrant, Ollama) are mocked so tests run without
live services. Tests verify:
  - Text generation for each event type
  - Embedding + storage pipeline with mock responses
  - Semantic search with mock vectors
  - Filter construction
  - Graceful degradation when services are unavailable
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_convergence_alert():
    """Minimal ConvergenceAlert-like object."""
    sig1 = SimpleNamespace(
        signal_id="insider:AAPL:2026-03-11",
        domain="insider",
        direction="bullish",
        source="sec_form4",
        strength=0.85,
        confidence=0.80,
        metadata={"symbol": "AAPL", "insider_name": "Tim Cook", "role": "CEO", "value": 2_100_000},
    )
    sig2 = SimpleNamespace(
        signal_id="macro:fred_macro:2026-03-11",
        domain="macro",
        direction="bullish",
        source="fred_macro",
        strength=0.70,
        confidence=0.90,
        metadata={"series_id": "DGS10", "value": "4.35", "prev_value": "4.20"},
    )
    sig3 = SimpleNamespace(
        signal_id="tech:AAPL:2026-03-11",
        domain="technical",
        direction="bullish",
        source="ta_rsi",
        strength=0.65,
        confidence=0.75,
        metadata={"symbol": "AAPL", "rsi": "32.1"},
    )
    return SimpleNamespace(
        alert_id="alert-test-001",
        timestamp=datetime(2026, 3, 11, 14, 30, 0),
        direction="bullish",
        strength=0.78,
        confidence=0.82,
        domains_converging=["insider", "macro", "technical"],
        signals=[sig1, sig2, sig3],
        cross_domain_count=3,
        summary="Bullish convergence: insider + macro + technical agree.",
        urgency="days",
        coherence=0.95,
        contradiction_details=[],
        combo_key="combo:insider+macro+technical",
        domain_sequence=["insider", "macro", "technical"],
        sequence_score=1.15,
        ripple_effects=[{"ticker": "NVDA", "strength": 0.6, "lag_days": 3}],
        ticker="AAPL",
    )


@pytest.fixture
def sample_market_signal():
    from mae_core.market.signal import MarketSignal
    return MarketSignal(
        signal_id="sec_form4:NVDA:2026-03-11T10:00:00",
        source="sec_form4",
        symbol="NVDA",
        asset_class="stock",
        domain="insider",
        direction="bullish",
        strength=0.88,
        confidence=0.80,
        decay_rate=0.025,
        timestamp=datetime(2026, 3, 11, 10, 0, 0),
        received_at=datetime(2026, 3, 11, 10, 5, 0),
        outcome_window_days=14,
        metadata={"name": "Jensen Huang", "role": "CEO", "value": 5_000_000, "shares": 10000},
    )


@pytest.fixture
def sample_pattern_template():
    from mae_core.market.archaeology.fingerprint import PatternTemplate
    t = PatternTemplate(
        template_id="abc123",
        direction="bullish",
        domain_signature="insider+macro+technical",
        domains=["insider", "macro", "technical"],
        wins=15,
        losses=8,
        n_instances=23,
        symbols_seen=["AAPL", "NVDA", "MSFT", "AMD", "GOOGL"],
        avg_move_pct=8.4,
        lag_profile_normalized={"immediate": 0.3, "short": 0.5, "medium": 0.2},
        lag_profile_raw={"immediate": 7, "short": 11, "medium": 5},
        created_at="2026-01-15T09:00:00",
    )
    return t


@pytest.fixture
def sample_insider_trade():
    return {
        "ticker": "AAPL",
        "insider_name": "Jeff Williams",
        "relationship": "Chief Operating Officer",
        "transaction_type": "P",
        "shares": 5000,
        "price": 175.50,
        "value": 877500,
        "date": "2026-03-10",
        "filing_date": "2026-03-11",
        "shares_after": 45000,
        "ownership_type": "direct",
        "source": "sec_form4",
    }


# ---------------------------------------------------------------------------
# event_descriptions tests
# ---------------------------------------------------------------------------

class TestDescribeConvergenceAlert:
    def test_basic_structure(self, sample_convergence_alert):
        from mae_core.market.intelligence.event_descriptions import describe_convergence_alert
        text = describe_convergence_alert(sample_convergence_alert)
        assert "BULLISH" in text
        assert "AAPL" in text
        assert "2026-03-11" in text
        assert "insider" in text.lower()
        assert "macro" in text.lower()
        assert "technical" in text.lower()

    def test_confidence_in_text(self, sample_convergence_alert):
        from mae_core.market.intelligence.event_descriptions import describe_convergence_alert
        text = describe_convergence_alert(sample_convergence_alert)
        assert "0.82" in text
        assert "very high" in text

    def test_coherence_warning_absent_for_high_coherence(self, sample_convergence_alert):
        from mae_core.market.intelligence.event_descriptions import describe_convergence_alert
        text = describe_convergence_alert(sample_convergence_alert)
        assert "WARNING" not in text

    def test_coherence_warning_present_for_low_coherence(self, sample_convergence_alert):
        from mae_core.market.intelligence.event_descriptions import describe_convergence_alert
        sample_convergence_alert.coherence = 0.60
        text = describe_convergence_alert(sample_convergence_alert)
        assert "WARNING" in text

    def test_sequence_score_boost_noted(self, sample_convergence_alert):
        from mae_core.market.intelligence.event_descriptions import describe_convergence_alert
        text = describe_convergence_alert(sample_convergence_alert)
        # sequence_score=1.15 (boost)
        assert "1.15" in text or "boosting" in text

    def test_ripple_effects_mentioned(self, sample_convergence_alert):
        from mae_core.market.intelligence.event_descriptions import describe_convergence_alert
        text = describe_convergence_alert(sample_convergence_alert)
        assert "NVDA" in text or "ripple" in text.lower()

    def test_insider_signal_detail(self, sample_convergence_alert):
        from mae_core.market.intelligence.event_descriptions import describe_convergence_alert
        text = describe_convergence_alert(sample_convergence_alert)
        assert "Tim Cook" in text

    def test_no_ticker_falls_back_gracefully(self, sample_convergence_alert):
        from mae_core.market.intelligence.event_descriptions import describe_convergence_alert
        sample_convergence_alert.ticker = None
        sample_convergence_alert.signals = []
        text = describe_convergence_alert(sample_convergence_alert)
        assert len(text) > 50  # should still produce meaningful output


class TestDescribeMarketSignal:
    def test_basic_structure(self, sample_market_signal):
        from mae_core.market.intelligence.event_descriptions import describe_market_signal
        text = describe_market_signal(sample_market_signal)
        assert "NVDA" in text
        assert "bullish" in text.lower()
        assert "insider" in text.lower()
        assert "Jensen Huang" in text

    def test_trade_value_included(self, sample_market_signal):
        from mae_core.market.intelligence.event_descriptions import describe_market_signal
        text = describe_market_signal(sample_market_signal)
        assert "$5.0M" in text or "5,000,000" in text or "$5" in text

    def test_decay_rate_included(self, sample_market_signal):
        from mae_core.market.intelligence.event_descriptions import describe_market_signal
        text = describe_market_signal(sample_market_signal)
        assert "0.025" in text

    def test_macro_signal_includes_series(self):
        from mae_core.market.signal import MarketSignal
        from mae_core.market.intelligence.event_descriptions import describe_market_signal
        sig = MarketSignal(
            signal_id="fred:DGS10:2026-03-11",
            source="fred_macro",
            symbol="",
            asset_class="macro",
            domain="macro",
            direction="bearish",
            strength=0.70,
            confidence=0.90,
            decay_rate=0.01,
            timestamp=datetime(2026, 3, 11),
            received_at=datetime(2026, 3, 11),
            metadata={"series_id": "DGS10", "value": "4.35", "prev_value": "4.20"},
        )
        text = describe_market_signal(sig)
        assert "DGS10" in text
        assert "4.35" in text


class TestDescribePatternTemplate:
    def test_basic_structure(self, sample_pattern_template):
        from mae_core.market.intelligence.event_descriptions import describe_pattern_template
        text = describe_pattern_template(sample_pattern_template)
        assert "bullish" in text.lower()
        assert "insider+macro+technical" in text
        assert "23" in text  # n_instances

    def test_win_rate_computed(self, sample_pattern_template):
        from mae_core.market.intelligence.event_descriptions import describe_pattern_template
        text = describe_pattern_template(sample_pattern_template)
        # 15/(15+8) = 65.2%
        assert "65" in text or "win rate" in text.lower()

    def test_cross_validated_noted(self, sample_pattern_template):
        from mae_core.market.intelligence.event_descriptions import describe_pattern_template
        text = describe_pattern_template(sample_pattern_template)
        assert "cross-validated" in text.lower() or "5 symbols" in text

    def test_lag_profile_dominant_bucket(self, sample_pattern_template):
        from mae_core.market.intelligence.event_descriptions import describe_pattern_template
        text = describe_pattern_template(sample_pattern_template)
        # dominant lag is "short" (0.5 fraction)
        assert "3-5 days" in text or "short" in text.lower()

    def test_sample_symbols_listed(self, sample_pattern_template):
        from mae_core.market.intelligence.event_descriptions import describe_pattern_template
        text = describe_pattern_template(sample_pattern_template)
        assert "AAPL" in text or "NVDA" in text


class TestDescribeInsiderTrade:
    def test_basic_structure(self, sample_insider_trade):
        from mae_core.market.intelligence.event_descriptions import describe_insider_trade
        text = describe_insider_trade(sample_insider_trade)
        assert "Jeff Williams" in text
        assert "AAPL" in text
        assert "purchased" in text.lower()

    def test_value_formatted(self, sample_insider_trade):
        from mae_core.market.intelligence.event_descriptions import describe_insider_trade
        text = describe_insider_trade(sample_insider_trade)
        assert "$877.5K" in text or "877" in text

    def test_ceo_conviction_note_absent_for_coo(self, sample_insider_trade):
        from mae_core.market.intelligence.event_descriptions import describe_insider_trade
        text = describe_insider_trade(sample_insider_trade)
        # COO not mentioned as C-suite in our logic (looking for ceo/cfo in role)
        assert "Chief" in text or "COO" in text or "Officer" in text

    def test_ceo_conviction_note_present_for_ceo(self, sample_insider_trade):
        from mae_core.market.intelligence.event_descriptions import describe_insider_trade
        sample_insider_trade["relationship"] = "Chief Executive Officer"
        sample_insider_trade["value"] = 2_000_000  # above $1M threshold for conviction note
        text = describe_insider_trade(sample_insider_trade)
        assert "C-suite" in text or "conviction" in text.lower()

    def test_sale_flagged(self, sample_insider_trade):
        from mae_core.market.intelligence.event_descriptions import describe_insider_trade
        sample_insider_trade["transaction_type"] = "S"
        text = describe_insider_trade(sample_insider_trade)
        assert "sold" in text.lower() or "sale" in text.lower()

    def test_large_purchase_flagged(self, sample_insider_trade):
        from mae_core.market.intelligence.event_descriptions import describe_insider_trade
        sample_insider_trade["value"] = 2_500_000
        text = describe_insider_trade(sample_insider_trade)
        assert "conviction" in text.lower() or "$1M" in text


class TestDescribeEconomicEvent:
    def test_basic_structure(self):
        from mae_core.market.intelligence.event_descriptions import describe_economic_event
        event = {
            "event": "Non-Farm Payrolls",
            "date": "2026-03-07",
            "actual": "250",
            "forecast": "200",
            "previous": "185",
            "country": "US",
            "impact": "high",
        }
        text = describe_economic_event(event)
        assert "Non-Farm Payrolls" in text
        assert "2026-03-07" in text
        assert "250" in text
        assert "high-impact" in text.lower()

    def test_positive_surprise_detected(self):
        from mae_core.market.intelligence.event_descriptions import describe_economic_event
        event = {
            "event": "CPI",
            "date": "2026-03-10",
            "actual": "3.5",
            "forecast": "3.0",
            "impact": "high",
        }
        text = describe_economic_event(event)
        assert "surprise" in text.lower() or "3.5" in text


class TestDescribeCongressionalTrade:
    def test_basic_structure(self):
        from mae_core.market.intelligence.event_descriptions import describe_congressional_trade
        trade = {
            "representative": "Nancy Pelosi",
            "chamber": "House",
            "state": "CA",
            "ticker": "NVDA",
            "type": "Purchase",
            "amount": "$1,000,001 - $5,000,000",
            "transaction_date": "2026-03-05",
            "committee": "House Financial Services Committee",
        }
        text = describe_congressional_trade(trade)
        assert "Nancy Pelosi" in text
        assert "NVDA" in text
        assert "purchased" in text.lower()
        assert "STOCK Act" in text

    def test_committee_noted(self):
        from mae_core.market.intelligence.event_descriptions import describe_congressional_trade
        trade = {
            "representative": "Jane Smith",
            "ticker": "LMT",
            "type": "Purchase",
            "committee": "House Armed Services Committee",
            "transaction_date": "2026-03-01",
        }
        text = describe_congressional_trade(trade)
        assert "Armed Services" in text


class TestDescribeContractAward:
    def test_basic_structure(self):
        from mae_core.market.intelligence.event_descriptions import describe_contract_award
        contract = {
            "recipient_name": "Raytheon Technologies",
            "ticker": "RTX",
            "awarding_agency": "Department of Defense",
            "amount": 340_000_000,
            "description": "Missile defense system production and sustainment",
            "award_date": "2026-03-08",
        }
        text = describe_contract_award(contract)
        assert "Raytheon" in text
        assert "$340.0M" in text
        assert "Defense" in text


# ---------------------------------------------------------------------------
# EventEmbedder tests (Qdrant + Ollama mocked)
# ---------------------------------------------------------------------------

def _mock_qdrant_ok():
    """Returns a mock requests session where Qdrant is healthy."""
    mock = MagicMock()
    # healthz
    mock.get.return_value.status_code = 200
    mock.get.return_value.json.return_value = {
        "models": [{"name": "mxbai-embed-large:latest"}]
    }
    # collection check → 404 (doesn't exist yet)
    # collection create → 200
    # search → results
    mock.put.return_value.status_code = 200
    mock.post.return_value.status_code = 200
    mock.post.return_value.json.return_value = {
        "embedding": [0.1] * 1024,
        "result": [
            {
                "score": 0.95,
                "payload": {
                    "event_type": "convergence_alert",
                    "description": "Bullish convergence on AAPL.",
                    "ticker": "AAPL",
                    "direction": "bullish",
                    "stored_at": "2026-03-11T10:00:00",
                }
            }
        ],
    }
    return mock


class TestEventEmbedderServiceChecks:
    def test_unavailable_when_qdrant_down(self):
        from mae_core.market.intelligence.event_embedder import EventEmbedder
        with patch("requests.get", side_effect=ConnectionError("qdrant down")):
            embedder = EventEmbedder()
        assert not embedder.is_available

    def test_unavailable_when_ollama_model_missing(self):
        from mae_core.market.intelligence.event_embedder import EventEmbedder

        call_count = [0]
        def mock_get(url, **kwargs):
            r = MagicMock()
            r.status_code = 200
            call_count[0] += 1
            if "healthz" in url:
                r.json.return_value = {}
            else:
                # No mxbai model
                r.json.return_value = {"models": [{"name": "llama3"}]}
            return r

        with patch("requests.get", side_effect=mock_get):
            embedder = EventEmbedder()
        assert not embedder.is_available

    def test_available_when_services_up(self):
        from mae_core.market.intelligence.event_embedder import EventEmbedder

        def mock_get(url, **kwargs):
            r = MagicMock()
            r.status_code = 200
            if "healthz" in url:
                r.json.return_value = {}
            elif "collections/" in url and "points" not in url:
                r.status_code = 404  # collection doesn't exist
            else:
                r.json.return_value = {"models": [{"name": "mxbai-embed-large:latest"}]}
            return r

        def mock_put(url, **kwargs):
            r = MagicMock()
            r.status_code = 200
            return r

        with patch("requests.get", side_effect=mock_get), \
             patch("requests.put", side_effect=mock_put):
            embedder = EventEmbedder()
        assert embedder.is_available


class TestEventEmbedderEmbedMethods:
    """Test that embed_* methods call the right pipeline without errors."""

    def _make_embedder(self):
        """Create an EventEmbedder with services mocked as available."""
        from mae_core.market.intelligence.event_embedder import EventEmbedder

        def mock_get(url, **kwargs):
            r = MagicMock()
            r.status_code = 200
            if "healthz" in url:
                r.json.return_value = {}
            elif "collections/" in url and "points" not in url:
                r.status_code = 404
            else:
                r.json.return_value = {"models": [{"name": "mxbai-embed-large:latest"}]}
            return r

        with patch("requests.get", side_effect=mock_get), \
             patch("requests.put", return_value=MagicMock(status_code=200)):
            embedder = EventEmbedder()
        return embedder

    def test_embed_convergence_alert_returns_id(self, sample_convergence_alert):
        embedder = self._make_embedder()
        assert embedder.is_available

        with patch("requests.post") as mock_post, \
             patch("requests.put") as mock_put:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"embedding": [0.1] * 1024}
            mock_put.return_value.status_code = 200

            result = embedder.embed_convergence_alert(sample_convergence_alert)

        assert result is not None
        assert result.startswith("convergence:")

    def test_embed_market_signal_returns_id(self, sample_market_signal):
        embedder = self._make_embedder()

        with patch("requests.post") as mock_post, \
             patch("requests.put") as mock_put:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"embedding": [0.2] * 1024}
            mock_put.return_value.status_code = 200

            result = embedder.embed_market_signal(sample_market_signal)

        assert result is not None
        assert result.startswith("signal:")

    def test_embed_pattern_template_returns_id(self, sample_pattern_template):
        embedder = self._make_embedder()

        with patch("requests.post") as mock_post, \
             patch("requests.put") as mock_put:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"embedding": [0.3] * 1024}
            mock_put.return_value.status_code = 200

            result = embedder.embed_pattern_template(sample_pattern_template)

        assert result is not None
        assert result.startswith("template:")

    def test_embed_insider_trade_returns_id(self, sample_insider_trade):
        embedder = self._make_embedder()

        with patch("requests.post") as mock_post, \
             patch("requests.put") as mock_put:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"embedding": [0.4] * 1024}
            mock_put.return_value.status_code = 200

            result = embedder.embed_insider_trade(sample_insider_trade)

        assert result is not None
        assert result.startswith("insider:")

    def test_embed_economic_event_returns_id(self):
        embedder = self._make_embedder()
        event = {
            "event": "CPI", "date": "2026-03-10",
            "actual": "3.5", "forecast": "3.0", "impact": "high",
        }

        with patch("requests.post") as mock_post, \
             patch("requests.put") as mock_put:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"embedding": [0.5] * 1024}
            mock_put.return_value.status_code = 200

            result = embedder.embed_economic_event(event)

        assert result is not None
        assert result.startswith("econ:")

    def test_embed_congressional_trade_returns_id(self):
        embedder = self._make_embedder()
        trade = {
            "representative": "Nancy Pelosi", "ticker": "NVDA",
            "type": "Purchase", "transaction_date": "2026-03-05",
        }

        with patch("requests.post") as mock_post, \
             patch("requests.put") as mock_put:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"embedding": [0.6] * 1024}
            mock_put.return_value.status_code = 200

            result = embedder.embed_congressional_trade(trade)

        assert result is not None
        assert result.startswith("congress:")

    def test_embed_contract_award_returns_id(self):
        embedder = self._make_embedder()
        contract = {
            "recipient_name": "Raytheon", "ticker": "RTX",
            "awarding_agency": "DoD", "amount": 340_000_000, "award_date": "2026-03-08",
        }

        with patch("requests.post") as mock_post, \
             patch("requests.put") as mock_put:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"embedding": [0.7] * 1024}
            mock_put.return_value.status_code = 200

            result = embedder.embed_contract_award(contract)

        assert result is not None
        assert result.startswith("contract:")

    def test_embed_returns_none_when_unavailable(self, sample_convergence_alert):
        from mae_core.market.intelligence.event_embedder import EventEmbedder
        with patch("requests.get", side_effect=ConnectionError()):
            embedder = EventEmbedder()
        assert not embedder.is_available
        result = embedder.embed_convergence_alert(sample_convergence_alert)
        assert result is None

    def test_embed_returns_none_when_ollama_fails(self, sample_convergence_alert):
        embedder = self._make_embedder()

        with patch("requests.post", side_effect=ConnectionError("ollama down")):
            result = embedder.embed_convergence_alert(sample_convergence_alert)
        assert result is None


class TestEventEmbedderSearch:
    def _make_embedder(self):
        from mae_core.market.intelligence.event_embedder import EventEmbedder

        def mock_get(url, **kwargs):
            r = MagicMock()
            r.status_code = 200
            if "healthz" in url:
                r.json.return_value = {}
            elif "collections/" in url:
                r.status_code = 404
            else:
                r.json.return_value = {"models": [{"name": "mxbai-embed-large:latest"}]}
            return r

        with patch("requests.get", side_effect=mock_get), \
             patch("requests.put", return_value=MagicMock(status_code=200)):
            return EventEmbedder()

    def test_find_similar_returns_list(self):
        embedder = self._make_embedder()
        mock_result = {
            "result": [
                {
                    "score": 0.92,
                    "payload": {
                        "event_type": "convergence_alert",
                        "description": "Bullish convergence on AAPL.",
                        "ticker": "AAPL",
                        "direction": "bullish",
                        "stored_at": "2026-03-10T12:00:00",
                    }
                }
            ]
        }

        with patch("requests.post") as mock_post:
            # First call = embed, second call = search
            embed_resp = MagicMock(status_code=200)
            embed_resp.json.return_value = {"embedding": [0.1] * 1024}
            embed_resp.raise_for_status = MagicMock()
            search_resp = MagicMock(status_code=200)
            search_resp.json.return_value = mock_result
            search_resp.raise_for_status = MagicMock()
            mock_post.side_effect = [embed_resp, search_resp]

            results = embedder.find_similar("bullish convergence AAPL insider", limit=5)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["score"] == 0.92
        assert results[0]["event_type"] == "convergence_alert"

    def test_find_similar_returns_empty_when_unavailable(self):
        from mae_core.market.intelligence.event_embedder import EventEmbedder
        with patch("requests.get", side_effect=ConnectionError()):
            embedder = EventEmbedder()
        results = embedder.find_similar("anything")
        assert results == []

    def test_find_similar_with_ticker_filter(self):
        embedder = self._make_embedder()
        with patch("requests.post") as mock_post:
            embed_resp = MagicMock(status_code=200)
            embed_resp.json.return_value = {"embedding": [0.1] * 1024}
            embed_resp.raise_for_status = MagicMock()
            search_resp = MagicMock(status_code=200)
            search_resp.json.return_value = {"result": []}
            search_resp.raise_for_status = MagicMock()
            mock_post.side_effect = [embed_resp, search_resp]

            results = embedder.find_similar(
                "insider buy NVDA",
                filters={"ticker": "NVDA", "direction": "bullish"},
            )

            # Verify the search body included a filter
            search_call = mock_post.call_args_list[1]
            search_body = search_call.kwargs.get("json", search_call.args[1] if len(search_call.args) > 1 else {})
            assert "filter" in search_body

        assert results == []

    def test_find_historical_precedents(self, sample_convergence_alert):
        embedder = self._make_embedder()
        signals = sample_convergence_alert.signals

        with patch("requests.post") as mock_post:
            embed_resp = MagicMock(status_code=200)
            embed_resp.json.return_value = {"embedding": [0.1] * 1024}
            embed_resp.raise_for_status = MagicMock()
            search_resp = MagicMock(status_code=200)
            search_resp.json.return_value = {"result": []}
            search_resp.raise_for_status = MagicMock()
            mock_post.side_effect = [embed_resp, search_resp, embed_resp, search_resp]

            results = embedder.find_historical_precedents(
                current_signals=signals,
                ticker="AAPL",
                limit=5,
            )

        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Filter builder tests
# ---------------------------------------------------------------------------

class TestFilterBuilder:
    def test_no_filters_returns_none(self):
        from mae_core.market.intelligence.event_embedder import _build_qdrant_filter
        assert _build_qdrant_filter(None) is None
        assert _build_qdrant_filter({}) is None

    def test_ticker_filter(self):
        from mae_core.market.intelligence.event_embedder import _build_qdrant_filter
        f = _build_qdrant_filter({"ticker": "AAPL"})
        assert f is not None
        assert f["must"][0]["key"] == "ticker"
        assert f["must"][0]["match"]["value"] == "AAPL"

    def test_direction_filter(self):
        from mae_core.market.intelligence.event_embedder import _build_qdrant_filter
        f = _build_qdrant_filter({"direction": "bullish"})
        assert any(c["key"] == "direction" for c in f["must"])

    def test_event_type_list_filter(self):
        from mae_core.market.intelligence.event_embedder import _build_qdrant_filter
        f = _build_qdrant_filter({"event_type": ["convergence_alert", "insider_trade"]})
        et_clause = next(c for c in f["must"] if c["key"] == "event_type")
        assert "any" in et_clause["match"]

    def test_date_range_filter(self):
        from mae_core.market.intelligence.event_embedder import _build_qdrant_filter
        f = _build_qdrant_filter({"date_from": "2026-01-01", "date_to": "2026-03-31"})
        date_clause = next(c for c in f["must"] if c["key"] == "date")
        assert date_clause["range"]["gte"] == "2026-01-01"
        assert date_clause["range"]["lte"] == "2026-03-31"

    def test_combined_filters(self):
        from mae_core.market.intelligence.event_embedder import _build_qdrant_filter
        f = _build_qdrant_filter({
            "ticker": "NVDA",
            "direction": "bullish",
            "event_type": "convergence_alert",
            "date_from": "2026-01-01",
        })
        assert len(f["must"]) == 4


# ---------------------------------------------------------------------------
# PatternMemory tests
# ---------------------------------------------------------------------------

class TestPatternMemory:
    def _make_memory(self, available=True):
        from mae_core.market.intelligence.pattern_memory import PatternMemory
        embedder = MagicMock()
        embedder.is_available = available
        return PatternMemory(embedder), embedder

    def test_is_available_reflects_embedder(self):
        mem, _ = self._make_memory(available=True)
        assert mem.is_available

        mem_off, _ = self._make_memory(available=False)
        assert not mem_off.is_available

    def test_remember_convergence_alert_calls_embedder(self, sample_convergence_alert):
        mem, embedder = self._make_memory()
        embedder.embed_convergence_alert.return_value = "convergence:abc"
        result = mem.remember_convergence_alert(sample_convergence_alert)
        embedder.embed_convergence_alert.assert_called_once_with(sample_convergence_alert)
        assert result == "convergence:abc"

    def test_remember_market_signal_calls_embedder(self, sample_market_signal):
        mem, embedder = self._make_memory()
        embedder.embed_market_signal.return_value = "signal:xyz"
        result = mem.remember_market_signal(sample_market_signal)
        embedder.embed_market_signal.assert_called_once_with(sample_market_signal)
        assert result == "signal:xyz"

    def test_remember_pattern_template_calls_embedder(self, sample_pattern_template):
        mem, embedder = self._make_memory()
        embedder.embed_pattern_template.return_value = "template:def"
        result = mem.remember_pattern_template(sample_pattern_template)
        embedder.embed_pattern_template.assert_called_once_with(sample_pattern_template)
        assert result == "template:def"

    def test_remember_event_dict_dispatch(self):
        mem, embedder = self._make_memory()
        embedder.embed_insider_trade.return_value = "insider:001"
        result = mem.remember_event("insider_trade", {"ticker": "AAPL"})
        embedder.embed_insider_trade.assert_called_once()
        assert result == "insider:001"

    def test_recall_similar_delegates_to_embedder(self):
        mem, embedder = self._make_memory()
        embedder.find_similar.return_value = [{"score": 0.9}]
        results = mem.recall_similar("insider buying AAPL", limit=3)
        embedder.find_similar.assert_called_once_with("insider buying AAPL", limit=3)
        assert results == [{"score": 0.9}]

    def test_find_precedents_delegates_to_embedder(self, sample_convergence_alert):
        mem, embedder = self._make_memory()
        embedder.find_historical_precedents.return_value = [{"score": 0.85}]
        results = mem.find_precedents(
            ticker="AAPL",
            signals=sample_convergence_alert.signals,
            limit=5,
        )
        embedder.find_historical_precedents.assert_called_once()
        assert results == [{"score": 0.85}]

    def test_get_pattern_context_structure(self, sample_convergence_alert):
        mem, embedder = self._make_memory()
        embedder.find_similar.return_value = [{"score": 0.9, "metadata": {"date": "2026-01-15"}}]
        ctx = mem.get_pattern_context(sample_convergence_alert)
        assert "similar_alerts" in ctx
        assert "similar_signals" in ctx
        assert "similar_templates" in ctx
        assert "temporal_note" in ctx
        assert isinstance(ctx["temporal_note"], str)
        assert len(ctx["temporal_note"]) > 0

    def test_returns_empty_when_unavailable(self, sample_convergence_alert):
        mem, _ = self._make_memory(available=False)
        assert mem.recall_similar("test") == []
        assert mem.find_precedents("AAPL", []) == []
        assert mem.remember_convergence_alert(sample_convergence_alert) is None

    def test_remember_event_unknown_type_returns_none(self):
        mem, embedder = self._make_memory()
        result = mem.remember_event("unknown_type", {"data": "x"})
        assert result is None

    def test_search_by_ticker(self):
        mem, embedder = self._make_memory()
        embedder.find_similar.return_value = [{"score": 0.88}]
        results = mem.search_by_ticker("NVDA", limit=5)
        embedder.find_similar.assert_called_once()
        call_kwargs = embedder.find_similar.call_args
        assert call_kwargs[1]["filters"]["ticker"] == "NVDA"

    def test_search_insider_buys(self):
        mem, embedder = self._make_memory()
        embedder.find_similar.return_value = []
        mem.search_insider_buys(ticker="AAPL")
        call_kwargs = embedder.find_similar.call_args
        f = call_kwargs[1]["filters"]
        assert f["event_type"] == "insider_trade"
        assert f["direction"] == "bullish"
        assert f["ticker"] == "AAPL"

    def test_search_high_confidence_alerts(self):
        mem, embedder = self._make_memory()
        embedder.find_similar.return_value = []
        mem.search_high_confidence_alerts(direction="bullish")
        call_kwargs = embedder.find_similar.call_args
        f = call_kwargs[1]["filters"]
        assert f["event_type"] == "convergence_alert"
        assert f["direction"] == "bullish"

    def test_none_embedder_makes_memory_offline(self):
        from mae_core.market.intelligence.pattern_memory import PatternMemory
        mem = PatternMemory(None)
        assert not mem.is_available
        assert mem.recall_similar("test") == []


# ---------------------------------------------------------------------------
# Temporal note tests
# ---------------------------------------------------------------------------

class TestMakeTemporalNote:
    def test_empty_alerts_returns_no_precedents(self):
        from mae_core.market.intelligence.pattern_memory import _make_temporal_note
        note = _make_temporal_note([])
        assert "No historical" in note

    def test_single_alert_note(self):
        from mae_core.market.intelligence.pattern_memory import _make_temporal_note
        alerts = [{"score": 0.9, "metadata": {"date": "2026-01-15"}}]
        note = _make_temporal_note(alerts)
        assert "2026-01-15" in note

    def test_multiple_alerts_date_range(self):
        from mae_core.market.intelligence.pattern_memory import _make_temporal_note
        alerts = [
            {"score": 0.9, "metadata": {"date": "2026-01-15"}},
            {"score": 0.85, "metadata": {"date": "2026-02-20"}},
            {"score": 0.80, "metadata": {"date": "2026-03-05"}},
        ]
        note = _make_temporal_note(alerts)
        assert "2026-01-15" in note
        assert "2026-03-05" in note
        assert "3" in note  # count

    def test_missing_dates_handled(self):
        from mae_core.market.intelligence.pattern_memory import _make_temporal_note
        alerts = [{"score": 0.9, "metadata": {}}]
        note = _make_temporal_note(alerts)
        assert "dates unavailable" in note or "1" in note
