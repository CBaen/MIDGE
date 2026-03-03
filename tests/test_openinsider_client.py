"""Tests for mae_core.market.apis.openinsider_client.

All tests are self-contained — no real HTTP calls are made.
requests.Session.get is mocked to return synthetic HTML or raise exceptions.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.apis.openinsider_client import (
    ClusterBuy,
    InsiderPurchase,
    OpenInsiderClient,
    _RATE_LIMIT_SECONDS,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic HTML builders
# ---------------------------------------------------------------------------

_TABLE_HEADER = """
<table class="tinytable">
  <tr>
    <th>#</th><th>Filing Date</th><th>Trade Date</th><th>Ticker</th>
    <th>Company</th><th>Insider Name</th><th>Title</th><th>Trade Type</th>
    <th>Price</th><th>Qty</th><th>Owned</th><th>Delta Own</th><th>Value</th>
  </tr>
"""

_TABLE_FOOTER = "</table>"


def _make_row(
    filing_date="2026-03-01",
    trade_date="2026-02-28",
    ticker="AAPL",
    company="Apple Inc",
    insider="Tim Cook",
    title="CEO",
    trade_type="P - Purchase",
    price="$150.00",
    qty="+1,000",
    owned="10,000",
    delta="+11%",
    value="$150,000",
    ticker_as_link=True,
):
    ticker_cell = (
        f'<td><a href="/AAPL">{ticker}</a></td>'
        if ticker_as_link
        else f"<td>{ticker}</td>"
    )
    return (
        f"<tr>"
        f"<td>1</td>"
        f"<td>{filing_date}</td>"
        f"<td>{trade_date}</td>"
        f"{ticker_cell}"
        f"<td>{company}</td>"
        f"<td>{insider}</td>"
        f"<td>{title}</td>"
        f"<td>{trade_type}</td>"
        f"<td>{price}</td>"
        f"<td>{qty}</td>"
        f"<td>{owned}</td>"
        f"<td>{delta}</td>"
        f"<td>{value}</td>"
        f"</tr>"
    )


def _make_html(*rows):
    return f"<html><body>{_TABLE_HEADER}{''.join(rows)}{_TABLE_FOOTER}</body></html>"


def _mock_response(html: str, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = html
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Dataclass creation
# ---------------------------------------------------------------------------


class TestInsiderPurchaseDataclass:
    def test_creation_all_fields(self):
        p = InsiderPurchase(
            filing_date="2026-03-01",
            trade_date="2026-02-28",
            ticker="AAPL",
            company_name="Apple Inc",
            insider_name="Tim Cook",
            title="CEO",
            trade_type="P - Purchase",
            price=150.0,
            quantity=1000,
            owned=10000,
            delta_owned_pct=11.0,
            value=150000.0,
        )
        assert p.ticker == "AAPL"
        assert p.price == 150.0
        assert p.quantity == 1000
        assert p.delta_owned_pct == 11.0

    def test_zero_defaults_are_valid(self):
        p = InsiderPurchase(
            filing_date="",
            trade_date="",
            ticker="",
            company_name="",
            insider_name="",
            title="",
            trade_type="",
            price=0.0,
            quantity=0,
            owned=0,
            delta_owned_pct=0.0,
            value=0.0,
        )
        assert p.value == 0.0


class TestClusterBuyDataclass:
    def test_creation(self):
        purchases = [
            InsiderPurchase(
                filing_date="2026-03-01",
                trade_date="2026-02-28",
                ticker="MSFT",
                company_name="Microsoft",
                insider_name=f"Insider {i}",
                title="Director",
                trade_type="P - Purchase",
                price=400.0,
                quantity=100,
                owned=1000,
                delta_owned_pct=11.0,
                value=40000.0,
            )
            for i in range(3)
        ]
        cluster = ClusterBuy(
            ticker="MSFT",
            company_name="Microsoft",
            insider_count=3,
            total_value=120000.0,
            insiders=purchases,
            date_range="2026-02-26 to 2026-02-28",
        )
        assert cluster.insider_count == 3
        assert cluster.total_value == 120000.0
        assert len(cluster.insiders) == 3


# ---------------------------------------------------------------------------
# _parse_table
# ---------------------------------------------------------------------------


class TestParseTable:
    def setup_method(self):
        self.client = OpenInsiderClient()

    def _soup(self, html: str):
        from bs4 import BeautifulSoup

        return BeautifulSoup(html, "html.parser")

    def test_parses_single_row(self):
        html = _make_html(_make_row())
        results = self.client._parse_table(self._soup(html))
        assert len(results) == 1
        p = results[0]
        assert p.ticker == "AAPL"
        assert p.company_name == "Apple Inc"
        assert p.insider_name == "Tim Cook"
        assert p.title == "CEO"
        assert p.price == 150.0
        assert p.quantity == 1000
        assert p.owned == 10000
        assert p.delta_owned_pct == 11.0
        assert p.value == 150000.0

    def test_ticker_uppercased(self):
        html = _make_html(_make_row(ticker="aapl"))
        results = self.client._parse_table(self._soup(html))
        assert results[0].ticker == "AAPL"

    def test_ticker_without_link(self):
        html = _make_html(_make_row(ticker="NVDA", ticker_as_link=False))
        results = self.client._parse_table(self._soup(html))
        assert results[0].ticker == "NVDA"

    def test_dollar_signs_and_commas_stripped(self):
        html = _make_html(
            _make_row(price="$1,234.56", qty="+5,678", owned="100,000", value="$7,000,000")
        )
        results = self.client._parse_table(self._soup(html))
        p = results[0]
        assert p.price == pytest.approx(1234.56)
        assert p.quantity == 5678
        assert p.owned == 100000
        assert p.value == pytest.approx(7000000.0)

    def test_new_keyword_becomes_100_pct(self):
        html = _make_html(_make_row(delta="New"))
        results = self.client._parse_table(self._soup(html))
        assert results[0].delta_owned_pct == 100.0

    def test_plus_sign_stripped_from_delta(self):
        html = _make_html(_make_row(delta="+25%"))
        results = self.client._parse_table(self._soup(html))
        assert results[0].delta_owned_pct == 25.0

    def test_empty_table_returns_empty_list(self):
        html = "<html><body><table class='tinytable'></table></body></html>"
        results = self.client._parse_table(self._soup(html))
        assert results == []

    def test_no_table_returns_empty_list(self):
        html = "<html><body><p>No data</p></body></html>"
        results = self.client._parse_table(self._soup(html))
        assert results == []

    def test_malformed_row_skipped(self):
        # Row with only 5 cells — below the 12-cell minimum
        bad_row = "<tr><td>1</td><td>2026-03-01</td><td>a</td><td>b</td><td>c</td></tr>"
        good_row = _make_row()
        html = _make_html(bad_row, good_row)
        results = self.client._parse_table(self._soup(html))
        # Only the good row is returned
        assert len(results) == 1
        assert results[0].ticker == "AAPL"

    def test_non_numeric_price_skipped(self):
        # Corrupt price that can't be parsed as float
        html = _make_html(_make_row(price="N/A$$$$"))
        # Should either skip row or produce 0.0, not crash
        try:
            results = self.client._parse_table(self._soup(html))
            # If it didn't skip, price should be 0.0 (empty string path)
        except Exception as exc:
            pytest.fail(f"Parser raised unexpectedly: {exc}")

    def test_fallback_table_selector(self):
        # No 'tinytable' class — uses fallback <th> text detection
        html = (
            "<html><body>"
            "<table>"
            "<tr><th>Filing Date</th><th>Trade Date</th><th>Ticker</th>"
            "<th>Company</th><th>Insider</th><th>Title</th><th>Type</th>"
            "<th>Price</th><th>Qty</th><th>Owned</th><th>Delta</th><th>Value</th>"
            "</tr>"
            + _make_row(ticker="META")
            + "</table></body></html>"
        )
        results = self.client._parse_table(self._soup(html))
        assert len(results) == 1
        assert results[0].ticker == "META"

    def test_multiple_rows_parsed(self):
        html = _make_html(
            _make_row(ticker="AAPL", insider="Tim Cook", value="$150,000"),
            _make_row(ticker="MSFT", insider="Satya Nadella", value="$200,000"),
            _make_row(ticker="NVDA", insider="Jensen Huang", value="$500,000"),
        )
        results = self.client._parse_table(self._soup(html))
        assert len(results) == 3
        tickers = {r.ticker for r in results}
        assert tickers == {"AAPL", "MSFT", "NVDA"}


# ---------------------------------------------------------------------------
# get_recent_purchases
# ---------------------------------------------------------------------------


class TestGetRecentPurchases:
    def setup_method(self):
        self.client = OpenInsiderClient()
        # Pre-set last request time so rate limiter doesn't sleep in tests
        self.client._last_request_time = 0.0

    @patch("mae_core.market.apis.openinsider_client.time.sleep")
    def test_returns_parsed_purchases(self, mock_sleep):
        html = _make_html(_make_row(ticker="TSLA", value="$300,000"))
        with patch.object(self.client._session, "get", return_value=_mock_response(html)):
            results = self.client.get_recent_purchases(days=7)
        assert len(results) == 1
        assert results[0].ticker == "TSLA"

    @patch("mae_core.market.apis.openinsider_client.time.sleep")
    def test_returns_empty_on_http_error(self, mock_sleep):
        import requests as _requests
        with patch.object(
            self.client._session,
            "get",
            side_effect=_requests.RequestException("connection refused"),
        ):
            results = self.client.get_recent_purchases()
        assert results == []

    @patch("mae_core.market.apis.openinsider_client.time.sleep")
    def test_url_includes_days_param(self, mock_sleep):
        html = _make_html()
        mock_get = MagicMock(return_value=_mock_response(html))
        with patch.object(self.client._session, "get", mock_get):
            self.client.get_recent_purchases(days=14)
        called_url = mock_get.call_args[0][0]
        assert "fd=14" in called_url

    @patch("mae_core.market.apis.openinsider_client.time.sleep")
    def test_url_includes_purchase_only_param(self, mock_sleep):
        html = _make_html()
        mock_get = MagicMock(return_value=_mock_response(html))
        with patch.object(self.client._session, "get", mock_get):
            self.client.get_recent_purchases()
        called_url = mock_get.call_args[0][0]
        assert "lt=1" in called_url


# ---------------------------------------------------------------------------
# get_cluster_buys
# ---------------------------------------------------------------------------


class TestGetClusterBuys:
    def setup_method(self):
        self.client = OpenInsiderClient()

    def _patch_purchases(self, purchases):
        return patch.object(self.client, "get_recent_purchases", return_value=purchases)

    def _purchase(self, ticker, insider, value=50000.0, trade_date="2026-02-28"):
        return InsiderPurchase(
            filing_date="2026-03-01",
            trade_date=trade_date,
            ticker=ticker,
            company_name=f"{ticker} Corp",
            insider_name=insider,
            title="Director",
            trade_type="P - Purchase",
            price=100.0,
            quantity=int(value / 100),
            owned=10000,
            delta_owned_pct=5.0,
            value=value,
        )

    def test_three_insiders_forms_cluster(self):
        purchases = [
            self._purchase("XYZ", "Alice", 50000.0),
            self._purchase("XYZ", "Bob", 60000.0),
            self._purchase("XYZ", "Carol", 70000.0),
        ]
        with self._patch_purchases(purchases):
            clusters = self.client.get_cluster_buys(min_insiders=3)
        assert len(clusters) == 1
        assert clusters[0].ticker == "XYZ"
        assert clusters[0].insider_count == 3

    def test_two_insiders_does_not_form_cluster_at_min_3(self):
        purchases = [
            self._purchase("XYZ", "Alice"),
            self._purchase("XYZ", "Bob"),
        ]
        with self._patch_purchases(purchases):
            clusters = self.client.get_cluster_buys(min_insiders=3)
        assert clusters == []

    def test_duplicate_insider_name_deduplicated(self):
        # Same insider buying twice — should count as ONE unique insider
        purchases = [
            self._purchase("XYZ", "Alice", 50000.0, "2026-02-27"),
            self._purchase("XYZ", "Alice", 60000.0, "2026-02-28"),  # duplicate
            self._purchase("XYZ", "Bob", 70000.0),
            self._purchase("XYZ", "Carol", 80000.0),
        ]
        with self._patch_purchases(purchases):
            clusters = self.client.get_cluster_buys(min_insiders=3)
        assert len(clusters) == 1
        assert clusters[0].insider_count == 3  # Alice (deduped) + Bob + Carol

    def test_total_value_sums_unique_insiders(self):
        purchases = [
            self._purchase("XYZ", "Alice", 50000.0),
            self._purchase("XYZ", "Bob", 60000.0),
            self._purchase("XYZ", "Carol", 70000.0),
        ]
        with self._patch_purchases(purchases):
            clusters = self.client.get_cluster_buys(min_insiders=3)
        assert clusters[0].total_value == pytest.approx(180000.0)

    def test_sorted_by_total_value_descending(self):
        purchases = [
            self._purchase("SMALL", "A1", 10000.0),
            self._purchase("SMALL", "A2", 10000.0),
            self._purchase("SMALL", "A3", 10000.0),
            self._purchase("BIG", "B1", 200000.0),
            self._purchase("BIG", "B2", 300000.0),
            self._purchase("BIG", "B3", 400000.0),
        ]
        with self._patch_purchases(purchases):
            clusters = self.client.get_cluster_buys(min_insiders=3)
        assert len(clusters) == 2
        assert clusters[0].ticker == "BIG"
        assert clusters[1].ticker == "SMALL"

    def test_date_range_computed_correctly(self):
        purchases = [
            self._purchase("XYZ", "Alice", trade_date="2026-02-15"),
            self._purchase("XYZ", "Bob", trade_date="2026-02-20"),
            self._purchase("XYZ", "Carol", trade_date="2026-03-01"),
        ]
        with self._patch_purchases(purchases):
            clusters = self.client.get_cluster_buys(min_insiders=3)
        assert clusters[0].date_range == "2026-02-15 to 2026-03-01"

    def test_empty_purchases_returns_empty(self):
        with self._patch_purchases([]):
            clusters = self.client.get_cluster_buys()
        assert clusters == []


# ---------------------------------------------------------------------------
# Rate limiter behavior
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_rate_limiter_sleeps_when_called_too_soon(self):
        client = OpenInsiderClient()
        client._last_request_time = time.time()  # Simulate a very recent request

        with patch("mae_core.market.apis.openinsider_client.time.sleep") as mock_sleep:
            client._rate_limit()
            mock_sleep.assert_called_once()
            sleep_duration = mock_sleep.call_args[0][0]
            assert sleep_duration > 0
            assert sleep_duration <= _RATE_LIMIT_SECONDS

    def test_rate_limiter_does_not_sleep_when_enough_time_passed(self):
        client = OpenInsiderClient()
        # Simulate last request was 10 seconds ago — well beyond the 3-second limit
        client._last_request_time = time.time() - 10.0

        with patch("mae_core.market.apis.openinsider_client.time.sleep") as mock_sleep:
            client._rate_limit()
            mock_sleep.assert_not_called()

    def test_rate_limiter_updates_last_request_time(self):
        client = OpenInsiderClient()
        client._last_request_time = 0.0

        before = time.time()
        with patch("mae_core.market.apis.openinsider_client.time.sleep"):
            client._rate_limit()
        after = time.time()

        assert before <= client._last_request_time <= after
