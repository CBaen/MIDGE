"""Tests for EdgarEnhancedClient — 13F institutional holdings + SC 13D activist filings.

All HTTP calls are mocked. No real network requests are made.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.apis.edgar_enhanced_client import (
    ActivistFiling,
    EdgarEnhancedClient,
    InstitutionalHolding,
)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestInstitutionalHolding:
    def test_basic_construction(self):
        h = InstitutionalHolding(
            filer_name="Vanguard Group",
            filer_cik="0000102909",
            ticker="AAPL",
            company_name="Apple Inc.",
            shares=1_000_000,
            value_usd=175_000.0,
            filing_date="2025-11-15",
            period_of_report="2025-09-30",
        )
        assert h.filer_name == "Vanguard Group"
        assert h.ticker == "AAPL"
        assert h.shares == 1_000_000
        assert h.value_usd == 175_000.0

    def test_fields_are_accessible(self):
        h = InstitutionalHolding(
            filer_name="BlackRock",
            filer_cik="0001364742",
            ticker="MSFT",
            company_name="Microsoft Corp",
            shares=500_000,
            value_usd=200_000.0,
            filing_date="2025-11-14",
            period_of_report="2025-09-30",
        )
        assert h.filer_cik == "0001364742"
        assert h.company_name == "Microsoft Corp"
        assert h.period_of_report == "2025-09-30"


class TestActivistFiling:
    def test_basic_construction(self):
        f = ActivistFiling(
            filer_name="Carl Icahn",
            filer_cik="0000101821",
            subject_company="Target Corp",
            subject_ticker="TGT",
            filing_date="2025-12-01",
            form_type="SC 13D",
            percent_owned=5.7,
            purpose="Activist position (>5% ownership)",
        )
        assert f.filer_name == "Carl Icahn"
        assert f.form_type == "SC 13D"
        assert f.percent_owned == 5.7

    def test_amendment_form_type(self):
        f = ActivistFiling(
            filer_name="Bill Ackman",
            filer_cik="0001543160",
            subject_company="Howard Hughes Corp",
            subject_ticker="HHC",
            filing_date="2026-01-10",
            form_type="SC 13D/A",
            percent_owned=38.4,
            purpose="Amendment to existing position",
        )
        assert f.form_type == "SC 13D/A"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(json_data: dict, status_code: int = 200, raises=None):
    """Build a mock requests.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    if raises:
        mock_resp.raise_for_status.side_effect = raises
        mock_resp.json.side_effect = raises
    else:
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = json_data
    return mock_resp


_EFTS_13D_RESPONSE = {
    "hits": {
        "hits": [
            {
                "_source": {
                    "display_names": ["Icahn Associates Corp", "Target Corp"],
                    "entity_id": "101821",
                    "file_date": "2025-12-01",
                    "form_type": "SC 13D",
                }
            },
            {
                "_source": {
                    "display_names": ["Pershing Square Capital"],
                    "entity_id": "1543160",
                    "ciks": ["1543160"],
                    "file_date": "2025-11-28",
                    "form_type": "SC 13D/A",
                }
            },
        ]
    }
}

_EFTS_13F_RESPONSE = {
    "hits": {
        "hits": [
            {
                "_source": {
                    "display_names": ["Vanguard Group Inc"],
                    "entity_id": "102909",
                    "file_date": "2025-11-15",
                    "form_type": "13F-HR",
                    "period_of_report": "2025-09-30",
                }
            },
            {
                "_source": {
                    "entity_name": "BlackRock Inc",
                    "entity_id": "1364742",
                    "file_date": "2025-11-14",
                    "form_type": "13F-HR",
                    "period_of_report": "2025-09-30",
                }
            },
        ]
    }
}


# ---------------------------------------------------------------------------
# get_13d_filings tests
# ---------------------------------------------------------------------------

class TestGet13dFilings:
    def test_parses_nested_hits_format(self):
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(_EFTS_13D_RESPONSE)):
            results = client.get_13d_filings()

        assert len(results) == 2
        assert all(isinstance(r, ActivistFiling) for r in results)

    def test_first_filing_filer_name(self):
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(_EFTS_13D_RESPONSE)):
            results = client.get_13d_filings()
        assert results[0].filer_name == "Icahn Associates Corp"

    def test_first_filing_subject_company_from_display_names(self):
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(_EFTS_13D_RESPONSE)):
            results = client.get_13d_filings()
        # Second display_name is subject company
        assert results[0].subject_company == "Target Corp"

    def test_second_filing_no_subject_company(self):
        """Single display_name entry — subject_company should be empty string."""
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(_EFTS_13D_RESPONSE)):
            results = client.get_13d_filings()
        assert results[1].subject_company == ""

    def test_filing_date_parsed(self):
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(_EFTS_13D_RESPONSE)):
            results = client.get_13d_filings()
        assert results[0].filing_date == "2025-12-01"

    def test_purpose_default_value(self):
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(_EFTS_13D_RESPONSE)):
            results = client.get_13d_filings()
        assert results[0].purpose == "Activist position (>5% ownership)"

    def test_empty_response_returns_empty_list(self):
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response({})):
            results = client.get_13d_filings()
        assert results == []

    def test_malformed_entries_skipped(self):
        """Entries that raise during construction should be silently skipped."""
        bad_response = {
            "hits": {
                "hits": [
                    {"_source": None},  # source is None — attribute access will raise
                    {
                        "_source": {
                            "display_names": ["Good Filer"],
                            "entity_id": "999",
                            "file_date": "2025-12-05",
                            "form_type": "SC 13D",
                        }
                    },
                ]
            }
        }
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(bad_response)):
            results = client.get_13d_filings()
        # The bad entry should be skipped; at least the good one survives
        assert len(results) >= 1
        assert results[-1].filer_name == "Good Filer"

    def test_flat_hits_format_also_parsed(self):
        """Some EFTS responses return hits as a flat list, not nested dict."""
        flat_response = {
            "hits": [
                {
                    "display_names": ["Flat Filer Corp"],
                    "entity_id": "555",
                    "file_date": "2025-12-10",
                    "form_type": "SC 13D",
                }
            ]
        }
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(flat_response)):
            results = client.get_13d_filings()
        assert len(results) == 1
        assert results[0].filer_name == "Flat Filer Corp"


# ---------------------------------------------------------------------------
# get_recent_13f_filers tests
# ---------------------------------------------------------------------------

class TestGetRecent13fFilers:
    def test_parses_nested_hits_format(self):
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(_EFTS_13F_RESPONSE)):
            results = client.get_recent_13f_filers()

        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_filer_name_from_display_names(self):
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(_EFTS_13F_RESPONSE)):
            results = client.get_recent_13f_filers()
        assert results[0]["filer_name"] == "Vanguard Group Inc"

    def test_filer_name_falls_back_to_entity_name(self):
        """When display_names absent, entity_name is used."""
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(_EFTS_13F_RESPONSE)):
            results = client.get_recent_13f_filers()
        assert results[1]["filer_name"] == "BlackRock Inc"

    def test_cik_is_stringified(self):
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(_EFTS_13F_RESPONSE)):
            results = client.get_recent_13f_filers()
        assert results[0]["cik"] == "102909"

    def test_empty_response_returns_empty_list(self):
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response({})):
            results = client.get_recent_13f_filers()
        assert results == []


# ---------------------------------------------------------------------------
# search_filings tests
# ---------------------------------------------------------------------------

class TestSearchFilings:
    def test_returns_sources_from_hits(self):
        search_response = {
            "hits": {
                "hits": [
                    {"_source": {"form_type": "8-K", "entity_name": "Acme Corp"}},
                    {"_source": {"form_type": "8-K", "entity_name": "Beta Inc"}},
                ]
            }
        }
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(search_response)):
            results = client.search_filings("material event", forms="8-K")

        assert len(results) == 2
        assert results[0]["entity_name"] == "Acme Corp"

    def test_limit_respected(self):
        """Only up to `limit` results returned even if API sends more."""
        many_hits = {
            "hits": {
                "hits": [
                    {"_source": {"entity_name": f"Company {i}"}} for i in range(15)
                ]
            }
        }
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response(many_hits)):
            results = client.search_filings("test", limit=5)
        assert len(results) == 5

    def test_empty_response_returns_empty_list(self):
        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=_make_mock_response({})):
            results = client.search_filings("nothing")
        assert results == []


# ---------------------------------------------------------------------------
# Infrastructure / client behaviour tests
# ---------------------------------------------------------------------------

class TestClientInfrastructure:
    def test_user_agent_header_set(self):
        client = EdgarEnhancedClient()
        ua = client._session.headers.get("User-Agent", "")
        assert "MIDGE" in ua
        assert "@" in ua  # must contain email per SEC policy

    def test_accept_header_is_json(self):
        client = EdgarEnhancedClient()
        assert client._session.headers.get("Accept") == "application/json"

    def test_non_json_response_returns_empty_dict(self):
        """If the server returns HTML or garbage, _get should return {}."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("No JSON")

        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", return_value=mock_resp):
            result = client._get("https://efts.sec.gov/LATEST/search-index")
        assert result == {}

    def test_request_exception_returns_empty_dict(self):
        """Network errors should be swallowed and return {}."""
        import requests as req

        client = EdgarEnhancedClient()
        with patch.object(client._session, "get", side_effect=req.RequestException("timeout")):
            result = client._get("https://efts.sec.gov/LATEST/search-index")
        assert result == {}

    def test_rate_limiter_records_call_times(self):
        """After N calls, _call_times deque should have entries."""
        client = EdgarEnhancedClient()
        mock_resp = _make_mock_response({})
        with patch.object(client._session, "get", return_value=mock_resp):
            for _ in range(3):
                client._get("https://efts.sec.gov/LATEST/search-index")
        assert len(client._call_times) == 3

    def test_rate_limiter_deque_max_length(self):
        """Deque should cap at _RATE_LIMIT (10) entries."""
        from mae_core.market.apis.edgar_enhanced_client import _RATE_LIMIT
        client = EdgarEnhancedClient()
        mock_resp = _make_mock_response({})
        with patch.object(client._session, "get", return_value=mock_resp):
            for _ in range(_RATE_LIMIT + 5):
                client._get("https://efts.sec.gov/LATEST/search-index")
        assert len(client._call_times) == _RATE_LIMIT
