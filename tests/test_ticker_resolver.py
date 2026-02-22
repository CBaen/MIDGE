"""Tests for mae_core.market.apis.ticker_resolver."""

import pytest
from mae_core.market.apis.ticker_resolver import resolve, resolve_batch, _normalize


class TestNormalize:
    """Test company name normalization."""

    def test_strips_corporation(self):
        assert _normalize("LOCKHEED MARTIN CORPORATION") == "LOCKHEED MARTIN"

    def test_strips_inc(self):
        assert _normalize("Palantir Technologies Inc") == "PALANTIR"

    def test_strips_llc(self):
        assert _normalize("Bechtel LLC") == "BECHTEL"

    def test_collapses_whitespace(self):
        assert _normalize("  Boeing   Company  ") == "BOEING"

    def test_case_insensitive(self):
        assert _normalize("lockheed martin corp") == "LOCKHEED MARTIN"


class TestResolve:
    """Test ticker resolution from company names."""

    def test_exact_match(self):
        assert resolve("Lockheed Martin") == "LMT"

    def test_with_corp_suffix(self):
        assert resolve("LOCKHEED MARTIN CORPORATION") == "LMT"

    def test_raytheon_technologies(self):
        assert resolve("Raytheon Technologies Corp") == "RTX"

    def test_boeing_company(self):
        assert resolve("The Boeing Company") == "BA"

    def test_booz_allen_hamilton(self):
        assert resolve("Booz Allen Hamilton Inc") == "BAH"

    def test_booz_allen_short(self):
        assert resolve("Booz Allen") == "BAH"

    def test_amazon_web_services(self):
        assert resolve("Amazon Web Services Inc") == "AMZN"

    def test_unknown_company(self):
        assert resolve("Totally Unknown Fake Company XYZ") is None

    def test_private_company_returns_none(self):
        # Bechtel is known-private — resolve returns None
        assert resolve("Bechtel") is None

    def test_case_insensitive_resolve(self):
        assert resolve("northrop grumman systems corp") == "NOC"

    def test_general_dynamics(self):
        assert resolve("General Dynamics Information Technology") == "GD"

    def test_l3harris(self):
        assert resolve("L3Harris Technologies") == "LHX"

    def test_pfizer(self):
        assert resolve("Pfizer Inc") == "PFE"

    def test_microsoft(self):
        assert resolve("Microsoft Corporation") == "MSFT"


class TestResolveBatch:
    """Test batch resolution."""

    def test_batch_multiple(self):
        results = resolve_batch(["Boeing", "Lockheed Martin Corp", "Unknown XYZ"])
        assert results["Boeing"] == "BA"
        assert results["Lockheed Martin Corp"] == "LMT"
        assert results["Unknown XYZ"] is None

    def test_batch_empty(self):
        assert resolve_batch([]) == {}
