"""
cot_client.py - CFTC Commitments of Traders (COT) Report Client

Fetches weekly COT data from the CFTC via the cot-reports library.
Tracks commercial/noncommercial/small trader net positions for futures.

When commercials flip to heavily long or short on a futures contract,
it's a regime-level signal. Commercial hedgers are the "smart money"
in futures markets — they know their own industry's supply/demand.

Data updates weekly (Fridays at 3:30 PM ET, covering Tuesday positions).
No API key required. Data source: CFTC.gov
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

REQUEST_DELAY = 2.0
CACHE_DURATION = 86400  # 24h — COT data only updates weekly

# Map MIDGE futures tickers to CFTC contract names used by cot-reports
TICKER_TO_CFTC = {
    "ES=F": "E-MINI S&P 500",
    "NQ=F": "NASDAQ-100",
    "YM=F": "MINI DOW JONES",
    "RTY=F": "E-MINI RUSSELL 2000",
    "GC=F": "GOLD",
    "CL=F": "CRUDE OIL, LIGHT SWEET",
    "SI=F": "SILVER",
    "ZB=F": "U.S. TREASURY BONDS",
    "ZN=F": "10-YEAR U.S. TREASURY NOTES",
    "NG=F": "NATURAL GAS",
}


@dataclass
class COTSignal:
    """CFTC Commitments of Traders positioning data for a futures contract."""

    ticker: str
    contract_name: str
    commercial_long: int
    commercial_short: int
    commercial_net: int
    noncommercial_long: int
    noncommercial_short: int
    noncommercial_net: int
    small_trader_net: int
    open_interest: int
    pct_commercial_long: float  # commercial_long / open_interest
    report_date: str            # YYYY-MM-DD

    signal_source: str = "cot_positioning"
    decay_rate: float = 0.03    # ~23 day half-life (weekly data, slow signal)
    confidence: float = 0.55

    def to_plain_language(self) -> str:
        comm_stance = "long" if self.commercial_net > 0 else "short"
        return (
            f"{self.ticker} ({self.contract_name}): Commercials net {comm_stance} "
            f"{abs(self.commercial_net):,} contracts, "
            f"OI {self.open_interest:,}, "
            f"report {self.report_date}"
        )


class COTClient:
    """
    Client for CFTC Commitments of Traders reports.

    Uses the cot-reports library to download and parse weekly COT data.
    No API key required. Data is cached for 24 hours (weekly updates only).
    """

    def __init__(self, provider=None):
        self._provider = provider
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MIDGE Trading Research"})
        self._last_request_time: float = 0.0
        self._cache: Optional[List[COTSignal]] = None
        self._cache_time: float = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def get_latest_positions(
        self, symbols: Optional[List[str]] = None
    ) -> List[COTSignal]:
        """
        Get latest COT positions for futures contracts.

        Args:
            symbols: List of MIDGE futures tickers (e.g. ["ES=F", "GC=F"]).
                     If None, returns all mapped contracts.

        Returns:
            List of COTSignal objects with net positioning data.
        """
        if self._cache is not None and time.time() - self._cache_time < CACHE_DURATION:
            signals = self._cache
            if symbols:
                signals = [s for s in signals if s.ticker in symbols]
            return signals

        self._rate_limit()

        try:
            import cot_reports as cot
        except ImportError:
            logger.warning("cot-reports library not installed. pip install cot-reports")
            return []

        try:
            # Fetch the most recent legacy futures-only report
            df = cot.cot_year(year=datetime.now().year, cot_report_type="legacy_fut")

            if df is None or df.empty:
                logger.warning("COT: no data returned for %d", datetime.now().year)
                return []

            signals = self._parse_dataframe(df, symbols)
            self._cache = signals
            self._cache_time = time.time()
            logger.debug("COT: fetched %d positions", len(signals))
            return signals

        except Exception as exc:
            logger.warning("COT fetch failed: %s", exc)
            return []

    def get_all_positions(
        self, symbols: Optional[List[str]] = None, years: Optional[List[int]] = None
    ) -> List[COTSignal]:
        """
        Get ALL weekly COT positions for futures contracts across multiple years.

        Unlike get_latest_positions() which returns only the most recent row per
        contract, this returns every weekly report — ideal for backfilling archives.

        Args:
            symbols: List of MIDGE futures tickers. If None, returns all mapped.
            years: List of years to fetch (e.g. [2025, 2026]). Defaults to current year.

        Returns:
            List of COTSignal objects, one per contract per weekly report.
        """
        self._rate_limit()

        try:
            import cot_reports as cot
        except ImportError:
            logger.warning("cot-reports library not installed. pip install cot-reports")
            return []

        if years is None:
            years = [datetime.now().year]

        all_signals = []
        for year in years:
            try:
                df = cot.cot_year(year=year, cot_report_type="legacy_fut")
                if df is None or df.empty:
                    logger.warning("COT: no data for year %d", year)
                    continue
                signals = self._parse_all_rows(df, symbols)
                all_signals.extend(signals)
                logger.info("COT: %d positions from year %d", len(signals), year)
            except Exception as exc:
                logger.warning("COT fetch failed for year %d: %s", year, exc)

        return all_signals

    def _parse_all_rows(
        self, df, symbols: Optional[List[str]] = None
    ) -> List[COTSignal]:
        """Parse ALL rows from cot-reports DataFrame (not just latest)."""
        results = []
        target_tickers = symbols or list(TICKER_TO_CFTC.keys())

        # Find column names
        name_col = None
        for col in ["Market and Exchange Names", "Market_and_Exchange_Names"]:
            if col in df.columns:
                name_col = col
                break
        if name_col is None:
            return results

        date_col = None
        for col in ["As of Date in Form YYYY-MM-DD", "As_of_Date_In_Form_YYYY-MM-DD", "Report_Date_as_YYYY-MM-DD"]:
            if col in df.columns:
                date_col = col
                break

        for ticker in target_tickers:
            cftc_name = TICKER_TO_CFTC.get(ticker)
            if not cftc_name:
                continue

            mask = df[name_col].str.contains(cftc_name, case=False, na=False)
            contract_df = df[mask]
            if contract_df.empty:
                continue

            for _, row in contract_df.iterrows():
                try:
                    def _get(col_variants, default=0):
                        for v in col_variants:
                            if v in row.index:
                                val = row[v]
                                try:
                                    return int(float(val))
                                except (ValueError, TypeError):
                                    continue
                        return default

                    comm_long = _get(["Comm_Positions_Long_All", "Commercial Long", "Comm_Positions_Long_Old"])
                    comm_short = _get(["Comm_Positions_Short_All", "Commercial Short", "Comm_Positions_Short_Old"])
                    noncomm_long = _get(["NonComm_Positions_Long_All", "Noncommercial Long", "NonComm_Positions-Long_All"])
                    noncomm_short = _get(["NonComm_Positions_Short_All", "Noncommercial Short", "NonComm_Positions-Short_All"])
                    oi = _get(["Open_Interest_All", "Open Interest", "Open_Interest_Old"], default=1)

                    comm_net = comm_long - comm_short
                    noncomm_net = noncomm_long - noncomm_short
                    small_net = oi - (comm_long + comm_short + noncomm_long + noncomm_short)
                    pct_comm = comm_long / max(1, oi)

                    report_date = str(row.get(date_col, "")) if date_col else ""
                    if hasattr(report_date, "strftime"):
                        report_date = report_date.strftime("%Y-%m-%d")

                    results.append(COTSignal(
                        ticker=ticker,
                        contract_name=cftc_name,
                        commercial_long=comm_long,
                        commercial_short=comm_short,
                        commercial_net=comm_net,
                        noncommercial_long=noncomm_long,
                        noncommercial_short=noncomm_short,
                        noncommercial_net=noncomm_net,
                        small_trader_net=small_net,
                        open_interest=oi,
                        pct_commercial_long=round(pct_comm, 4),
                        report_date=report_date[:10] if report_date else "",
                    ))
                except Exception as exc:
                    logger.debug("COT: failed to parse row for %s: %s", ticker, exc)

        return results

    def _parse_dataframe(
        self, df, symbols: Optional[List[str]] = None
    ) -> List[COTSignal]:
        """Parse cot-reports DataFrame into COTSignal objects."""
        results = []

        target_tickers = symbols or list(TICKER_TO_CFTC.keys())

        for ticker in target_tickers:
            cftc_name = TICKER_TO_CFTC.get(ticker)
            if not cftc_name:
                continue

            try:
                # Filter for the contract — match on Market and Sitename column
                name_col = None
                for col in ["Market and Exchange Names", "Market_and_Exchange_Names"]:
                    if col in df.columns:
                        name_col = col
                        break

                if name_col is None:
                    logger.debug("COT: cannot find name column in dataframe")
                    continue

                mask = df[name_col].str.contains(cftc_name, case=False, na=False)
                contract_df = df[mask]

                if contract_df.empty:
                    continue

                # Get the most recent row
                date_col = None
                for col in ["As of Date in Form YYYY-MM-DD", "As_of_Date_In_Form_YYYY-MM-DD", "Report_Date_as_YYYY-MM-DD"]:
                    if col in contract_df.columns:
                        date_col = col
                        break

                if date_col:
                    contract_df = contract_df.sort_values(date_col, ascending=False)

                row = contract_df.iloc[0]

                # Extract positioning columns
                def _get(col_variants, default=0):
                    for v in col_variants:
                        if v in row.index:
                            val = row[v]
                            try:
                                return int(float(val))
                            except (ValueError, TypeError):
                                continue
                    return default

                comm_long = _get(["Comm_Positions_Long_All", "Commercial Long", "Comm_Positions_Long_Old"])
                comm_short = _get(["Comm_Positions_Short_All", "Commercial Short", "Comm_Positions_Short_Old"])
                noncomm_long = _get(["NonComm_Positions_Long_All", "Noncommercial Long", "NonComm_Positions-Long_All"])
                noncomm_short = _get(["NonComm_Positions_Short_All", "Noncommercial Short", "NonComm_Positions-Short_All"])
                oi = _get(["Open_Interest_All", "Open Interest", "Open_Interest_Old"], default=1)

                comm_net = comm_long - comm_short
                noncomm_net = noncomm_long - noncomm_short
                small_net = oi - (comm_long + comm_short + noncomm_long + noncomm_short)

                pct_comm = comm_long / max(1, oi)

                report_date = str(row.get(date_col, "")) if date_col else ""
                if hasattr(report_date, "strftime"):
                    report_date = report_date.strftime("%Y-%m-%d")

                results.append(COTSignal(
                    ticker=ticker,
                    contract_name=cftc_name,
                    commercial_long=comm_long,
                    commercial_short=comm_short,
                    commercial_net=comm_net,
                    noncommercial_long=noncomm_long,
                    noncommercial_short=noncomm_short,
                    noncommercial_net=noncomm_net,
                    small_trader_net=small_net,
                    open_interest=oi,
                    pct_commercial_long=round(pct_comm, 4),
                    report_date=report_date[:10] if report_date else "",
                ))

            except Exception as exc:
                logger.debug("COT: failed to parse %s: %s", ticker, exc)
                continue

        return results


def get_latest_positions(symbols: Optional[List[str]] = None) -> List[COTSignal]:
    """Convenience function for one-shot COT data fetch."""
    return COTClient().get_latest_positions(symbols)


def get_all_positions(symbols: Optional[List[str]] = None, years: Optional[List[int]] = None) -> List[COTSignal]:
    """Convenience function for multi-year COT history."""
    return COTClient().get_all_positions(symbols, years)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Testing CFTC COT Reports...")

    client = COTClient()
    positions = client.get_latest_positions(["ES=F", "GC=F", "CL=F"])

    if positions:
        for p in positions:
            print(f"  {p.to_plain_language()}")
    else:
        print("  No data returned")

    print("\nDone.")
