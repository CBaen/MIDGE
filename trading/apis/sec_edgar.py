#!/usr/bin/env python3
"""
sec_edgar.py - SEC EDGAR Form 4 Scraper

Scrapes insider trading data from SEC EDGAR.
Form 4 must be filed within 2 business days of insider trades.

Free data source - no API key required.
Respects SEC's rate limiting (10 requests/second max).

Data Points Extracted:
- Who: Insider name, title, relationship to company
- What: Securities traded (stock symbol)
- When: Transaction date
- How Much: Number of shares, price per share
- Direction: Acquisition (buy) or Disposition (sell)
"""

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import requests

# SEC EDGAR API endpoints
SEC_BASE_URL = "https://data.sec.gov"
SEC_WWW_URL = "https://www.sec.gov"
SEC_SUBMISSIONS_URL = f"{SEC_BASE_URL}/submissions"
SEC_FILINGS_URL = f"{SEC_WWW_URL}/cgi-bin/browse-edgar"
SEC_TICKERS_URL = f"{SEC_WWW_URL}/files/company_tickers.json"

# User agent required by SEC (they block default requests user agents)
SEC_USER_AGENT = "MIDGE Trading Research contact@example.com"

# Rate limiting - SEC allows max 10 req/sec
REQUEST_DELAY = 0.15  # 150ms between requests


@dataclass
class InsiderTrade:
    """Single insider trading transaction from Form 4."""
    # Filer Information
    filer_name: str
    filer_title: str
    filer_relationship: str  # Director, Officer, 10% Owner, Other

    # Company Information
    company_name: str
    company_cik: str
    ticker_symbol: str

    # Transaction Details
    transaction_date: str
    transaction_type: str  # Acquisition (A) or Disposition (D)
    shares: float
    price_per_share: float
    total_value: float

    # Post-Transaction Holdings
    shares_owned_after: float

    # Filing Information
    filing_date: str
    accession_number: str
    form_type: str

    # Signal metadata for MIDGE
    signal_source: str = "insider"
    decay_rate: float = 0.05  # ~14 day half-life

    def to_plain_language(self) -> str:
        """Format for Guiding Light's dashboard."""
        action = "bought" if self.transaction_type == "A" else "sold"
        return (
            f"{self.filer_name} ({self.filer_title}) {action} "
            f"${self.total_value:,.0f} of {self.ticker_symbol or self.company_name} "
            f"({self.shares:,.0f} shares @ ${self.price_per_share:.2f})"
        )


class SECEdgarClient:
    """
    Client for SEC EDGAR API.

    Respects rate limits and uses required User-Agent header.
    """

    def __init__(self, user_agent: str = SEC_USER_AGENT):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate"
        })
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict = None) -> Optional[requests.Response]:
        """Make rate-limited GET request."""
        self._rate_limit()
        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response
            else:
                print(f"SEC EDGAR error: {response.status_code} for {url}")
                return None
        except Exception as e:
            print(f"SEC EDGAR request failed: {e}")
            return None

    def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK number for a ticker symbol.

        CIK is the Central Index Key - SEC's company identifier.
        """
        # SEC provides a ticker-to-CIK mapping file
        response = self._get(SEC_TICKERS_URL)

        if not response:
            return None

        try:
            data = response.json()
            ticker_upper = ticker.upper()

            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker_upper:
                    # CIK needs to be zero-padded to 10 digits
                    cik = str(entry.get("cik_str", ""))
                    return cik.zfill(10)

            return None
        except Exception as e:
            print(f"Error parsing CIK mapping: {e}")
            return None

    def get_company_filings(self, cik: str, form_type: str = "4") -> List[dict]:
        """
        Get recent filings for a company.

        Args:
            cik: Company CIK (10-digit, zero-padded)
            form_type: Form type to filter ("4" for insider trades)

        Returns:
            List of filing metadata
        """
        # SEC's submissions endpoint
        submissions_url = f"{SEC_SUBMISSIONS_URL}/CIK{cik}.json"
        response = self._get(submissions_url)

        if not response:
            return []

        try:
            data = response.json()
            filings = data.get("filings", {}).get("recent", {})

            # Extract relevant fields
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            primary_docs = filings.get("primaryDocument", [])

            results = []
            for i, form in enumerate(forms):
                if form == form_type:
                    # Build the full document URL
                    acc_no_dash = accessions[i].replace("-", "") if i < len(accessions) else ""
                    primary_doc = primary_docs[i] if i < len(primary_docs) else ""
                    cik_int = int(cik)  # Remove leading zeros for URL

                    results.append({
                        "form": form,
                        "filing_date": dates[i] if i < len(dates) else None,
                        "accession_number": accessions[i] if i < len(accessions) else None,
                        "primary_document": primary_doc,
                        "document_url": f"{SEC_WWW_URL}/Archives/edgar/data/{cik_int}/{acc_no_dash}/{primary_doc}",
                        "cik": cik,
                        "cik_int": cik_int,
                        "company_name": data.get("name", "")
                    })

            return results

        except Exception as e:
            print(f"Error parsing filings: {e}")
            return []

    def parse_form4(self, cik: str, accession_number: str, document_url: str = None) -> List[InsiderTrade]:
        """
        Parse a Form 4 filing to extract insider trades.

        SEC Form 4s are served as HTML (XSLT-rendered from XML).
        This parser handles both formats.

        Args:
            cik: Company CIK
            accession_number: Filing accession number
            document_url: Direct URL to the document (preferred)

        Returns:
            List of InsiderTrade objects
        """
        # Use provided URL or construct one
        if document_url:
            response = self._get(document_url)
        else:
            # Fallback - construct URL
            acc_clean = accession_number.replace("-", "")
            xml_url = f"{SEC_WWW_URL}/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession_number}.xml"
            response = self._get(xml_url)

        if not response:
            return []

        content = response.text

        # Check if it's HTML (XSLT-rendered) or raw XML
        if content.strip().startswith("<!DOCTYPE html") or "<html" in content[:500]:
            return self._parse_form4_html(content, cik, accession_number)

        # Try XML parsing
        try:
            root = ET.fromstring(response.content)

            # Namespace handling - Form 4 uses a specific namespace
            ns = {"": "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"}

            # Extract filer information
            reporting_owner = root.find(".//reportingOwner") or root.find("reportingOwner")

            filer_name = ""
            filer_title = ""
            filer_relationship = []

            if reporting_owner is not None:
                # Get owner name
                owner_id = reporting_owner.find("reportingOwnerId")
                if owner_id is not None:
                    name_elem = owner_id.find("rptOwnerName")
                    if name_elem is not None:
                        filer_name = name_elem.text or ""

                # Get relationship
                relationship = reporting_owner.find("reportingOwnerRelationship")
                if relationship is not None:
                    if relationship.find("isDirector") is not None:
                        is_dir = relationship.find("isDirector").text
                        if is_dir and is_dir.lower() in ("1", "true"):
                            filer_relationship.append("Director")

                    if relationship.find("isOfficer") is not None:
                        is_off = relationship.find("isOfficer").text
                        if is_off and is_off.lower() in ("1", "true"):
                            filer_relationship.append("Officer")
                            # Get officer title
                            title_elem = relationship.find("officerTitle")
                            if title_elem is not None:
                                filer_title = title_elem.text or ""

                    if relationship.find("isTenPercentOwner") is not None:
                        is_ten = relationship.find("isTenPercentOwner").text
                        if is_ten and is_ten.lower() in ("1", "true"):
                            filer_relationship.append("10% Owner")

            # Extract company information
            issuer = root.find(".//issuer") or root.find("issuer")
            company_name = ""
            ticker_symbol = ""
            company_cik = cik

            if issuer is not None:
                name_elem = issuer.find("issuerName")
                if name_elem is not None:
                    company_name = name_elem.text or ""

                ticker_elem = issuer.find("issuerTradingSymbol")
                if ticker_elem is not None:
                    ticker_symbol = ticker_elem.text or ""

                cik_elem = issuer.find("issuerCik")
                if cik_elem is not None:
                    company_cik = cik_elem.text or cik

            # Extract transactions
            trades = []

            # Non-derivative transactions (common stock)
            non_deriv_table = root.find(".//nonDerivativeTable") or root.find("nonDerivativeTable")
            if non_deriv_table is not None:
                for trans in non_deriv_table.findall("nonDerivativeTransaction"):
                    trade = self._parse_transaction(
                        trans,
                        filer_name=filer_name,
                        filer_title=filer_title,
                        filer_relationship=", ".join(filer_relationship) or "Other",
                        company_name=company_name,
                        company_cik=company_cik,
                        ticker_symbol=ticker_symbol,
                        accession_number=accession_number
                    )
                    if trade:
                        trades.append(trade)

            # Derivative transactions (options, etc.) - if needed
            # deriv_table = root.find(".//derivativeTable")
            # (More complex to parse - skip for now)

            return trades

        except ET.ParseError as e:
            print(f"XML parse error for {accession_number}: {e}")
            return []
        except Exception as e:
            print(f"Error parsing Form 4 {accession_number}: {e}")
            return []

    def _parse_form4_html(self, html_content: str, cik: str, accession_number: str) -> List[InsiderTrade]:
        """
        Parse Form 4 from HTML (XSLT-rendered) format.

        The SEC renders Form 4 XML files as HTML using XSLT.
        This extracts data from the HTML table structure.
        """
        import re

        trades = []

        try:
            # Extract key data using regex (more reliable than HTML parsing for SEC forms)
            # These forms have consistent structure

            # Filer name - in anchor tag after "Reporting Person"
            # Pattern: Reporting Person.*?<a href="...">NAME</a>
            filer_match = re.search(r'Reporting Person.*?<a[^>]*>([^<]+)</a>', html_content, re.DOTALL | re.IGNORECASE)
            filer_name = filer_match.group(1).strip() if filer_match else "Unknown"

            # Company name - look for issuer name
            company_match = re.search(r'Issuer Name.*?class="FormData[^"]*"[^>]*>([^<]+)', html_content, re.DOTALL | re.IGNORECASE)
            company_name = company_match.group(1).strip() if company_match else "Unknown"

            # Ticker symbol
            ticker_match = re.search(r'Trading Symbol.*?class="FormData[^"]*"[^>]*>([^<]+)', html_content, re.DOTALL | re.IGNORECASE)
            ticker_symbol = ticker_match.group(1).strip() if ticker_match else ""

            # Relationship checkboxes (Director, Officer, 10% Owner)
            relationships = []
            if re.search(r'Director.*?X|Director.*?checked', html_content, re.IGNORECASE):
                relationships.append("Director")
            if re.search(r'Officer.*?X|Officer.*?checked', html_content, re.IGNORECASE):
                relationships.append("Officer")
            if re.search(r'10%.*?Owner.*?X|10%.*?Owner.*?checked', html_content, re.IGNORECASE):
                relationships.append("10% Owner")

            filer_relationship = ", ".join(relationships) if relationships else "Other"

            # Officer title
            title_match = re.search(r'Officer.*?Title.*?class="FormData[^"]*"[^>]*>([^<]+)', html_content, re.DOTALL | re.IGNORECASE)
            filer_title = title_match.group(1).strip() if title_match else ""

            # Look for transaction rows in Table I (Non-Derivative Securities)
            # Pattern: Date | Title | Code | V | Shares | Price | D/A | Shares Owned
            trans_pattern = re.compile(
                r'(\d{2}/\d{2}/\d{4}).*?'  # Date
                r'class="FormData[^"]*"[^>]*>([^<]*)</.*?'  # Title of Security
                r'([PSMDGFC]).*?'  # Transaction Code
                r'([\d,\.]+).*?'  # Shares
                r'\$([\d,\.]+).*?'  # Price
                r'([AD])',  # Acquired/Disposed
                re.DOTALL
            )

            # Simplified extraction - look for transaction table data
            # Each row has date, security title, transaction code, shares, price, A/D

            # Find all monetary values that look like transactions
            price_pattern = re.compile(r'\$\s*([\d,]+\.?\d*)')
            share_pattern = re.compile(r'class="FormData[^"]*"[^>]*>([\d,]+)<')

            # Extract transaction data from table rows
            # Look for Table I section
            table1_match = re.search(r'Table I.*?</table>', html_content, re.DOTALL | re.IGNORECASE)
            if table1_match:
                table_content = table1_match.group(0)

                # Find transaction dates
                dates = re.findall(r'(\d{2}/\d{2}/\d{4})', table_content)

                # Find prices
                prices = [float(p.replace(',', '')) for p in price_pattern.findall(table_content)]

                # Find A or D codes
                ad_codes = re.findall(r'>([AD])<', table_content)

                # Find share amounts (large numbers)
                shares_list = [float(s.replace(',', '')) for s in re.findall(r'>([\d,]{1,15})<', table_content) if float(s.replace(',', '')) > 0]

                # Create trades from matched data
                if dates and prices:
                    for i, date in enumerate(dates[:len(prices)]):
                        price = prices[i] if i < len(prices) else 0
                        shares = shares_list[i] if i < len(shares_list) else 0
                        trans_type = ad_codes[i] if i < len(ad_codes) else "A"

                        if shares > 0 and price > 0:
                            trades.append(InsiderTrade(
                                filer_name=filer_name,
                                filer_title=filer_title,
                                filer_relationship=filer_relationship,
                                company_name=company_name,
                                company_cik=cik,
                                ticker_symbol=ticker_symbol,
                                transaction_date=date,
                                transaction_type=trans_type,
                                shares=shares,
                                price_per_share=price,
                                total_value=shares * price,
                                shares_owned_after=0,  # Not easily extracted from HTML
                                filing_date="",
                                accession_number=accession_number,
                                form_type="4"
                            ))

            # If no transactions found in table parsing, try simpler extraction
            if not trades:
                # Look for any transaction-like data
                # Format: date, shares, price
                all_dates = re.findall(r'(\d{2}/\d{2}/\d{4})', html_content)
                all_prices = price_pattern.findall(html_content)
                all_shares = [s for s in re.findall(r'>([\d,]+)<', html_content)
                             if s.replace(',', '').isdigit() and 100 < float(s.replace(',', '')) < 10000000]

                if all_dates and all_prices and all_shares:
                    # Take first transaction-like set
                    price = float(all_prices[0].replace(',', '')) if all_prices else 0
                    shares = float(all_shares[0].replace(',', '')) if all_shares else 0
                    date = all_dates[0] if all_dates else ""

                    if shares > 0 and price > 0:
                        # Determine if buy or sell from context
                        trans_type = "D" if re.search(r'Disposition|Sale|Sold', html_content, re.IGNORECASE) else "A"

                        trades.append(InsiderTrade(
                            filer_name=filer_name,
                            filer_title=filer_title,
                            filer_relationship=filer_relationship,
                            company_name=company_name,
                            company_cik=cik,
                            ticker_symbol=ticker_symbol,
                            transaction_date=date,
                            transaction_type=trans_type,
                            shares=shares,
                            price_per_share=price,
                            total_value=shares * price,
                            shares_owned_after=0,
                            filing_date="",
                            accession_number=accession_number,
                            form_type="4"
                        ))

        except Exception as e:
            print(f"Error parsing Form 4 HTML {accession_number}: {e}")

        return trades

    def _parse_transaction(self, trans_elem, **metadata) -> Optional[InsiderTrade]:
        """Parse a single transaction element from Form 4."""
        try:
            # Transaction date
            date_elem = trans_elem.find(".//transactionDate/value")
            trans_date = date_elem.text if date_elem is not None else None

            # Transaction amounts
            amounts = trans_elem.find("transactionAmounts")
            if amounts is None:
                return None

            shares_elem = amounts.find("transactionShares/value")
            shares = float(shares_elem.text) if shares_elem is not None and shares_elem.text else 0

            price_elem = amounts.find("transactionPricePerShare/value")
            price = float(price_elem.text) if price_elem is not None and price_elem.text else 0

            # Acquisition (A) or Disposition (D)
            code_elem = amounts.find("transactionAcquiredDisposedCode/value")
            trans_type = code_elem.text if code_elem is not None else "A"

            # Post-transaction holdings
            holdings = trans_elem.find("postTransactionAmounts")
            shares_after = 0
            if holdings is not None:
                after_elem = holdings.find("sharesOwnedFollowingTransaction/value")
                if after_elem is not None and after_elem.text:
                    shares_after = float(after_elem.text)

            # Skip if no meaningful transaction
            if shares == 0 or price == 0:
                return None

            return InsiderTrade(
                filer_name=metadata["filer_name"],
                filer_title=metadata["filer_title"],
                filer_relationship=metadata["filer_relationship"],
                company_name=metadata["company_name"],
                company_cik=metadata["company_cik"],
                ticker_symbol=metadata["ticker_symbol"],
                transaction_date=trans_date or "",
                transaction_type=trans_type,
                shares=shares,
                price_per_share=price,
                total_value=shares * price,
                shares_owned_after=shares_after,
                filing_date="",  # Will be set by caller
                accession_number=metadata["accession_number"],
                form_type="4"
            )

        except Exception as e:
            print(f"Error parsing transaction: {e}")
            return None


def get_recent_form4s(ticker: str, days: int = 30) -> List[InsiderTrade]:
    """
    Get recent Form 4 filings for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        days: Number of days to look back

    Returns:
        List of InsiderTrade objects
    """
    client = SECEdgarClient()

    # Get CIK for ticker
    cik = client.get_company_cik(ticker)
    if not cik:
        print(f"Could not find CIK for {ticker}")
        return []

    print(f"Found CIK {cik} for {ticker}")

    # Get recent Form 4 filings
    filings = client.get_company_filings(cik, form_type="4")
    print(f"Found {len(filings)} Form 4 filings")

    # Filter by date
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    recent_filings = [f for f in filings if f.get("filing_date", "") >= cutoff_date]
    print(f"Found {len(recent_filings)} filings in last {days} days")

    # Parse each filing
    all_trades = []
    for filing in recent_filings[:10]:  # Limit to 10 most recent
        trades = client.parse_form4(
            cik,
            filing["accession_number"],
            document_url=filing.get("document_url")
        )
        for trade in trades:
            trade.filing_date = filing.get("filing_date", "")
        all_trades.extend(trades)

    return all_trades


def search_politician_trades(politician_name: str = None) -> List[InsiderTrade]:
    """
    Search for trades by politicians (placeholder).

    Note: This would require cross-referencing SEC data with
    Congress member stock disclosures from other sources.

    For actual politician trade tracking, see:
    - https://www.capitoltrades.com (paid)
    - https://housestockwatcher.com (community)
    - Senate/House periodic transaction reports
    """
    # TODO: Implement politician trade lookup
    # This requires:
    # 1. Congress member list with associated companies
    # 2. Cross-reference with SEC Form 4 (if they're directors)
    # 3. Or separate tracking of STOCK Act disclosures
    print("Politician trade search requires additional data sources")
    return []


if __name__ == "__main__":
    # Test with Apple
    print("Testing SEC EDGAR Form 4 scraper...")
    print()

    trades = get_recent_form4s("AAPL", days=30)

    if trades:
        print(f"\nFound {len(trades)} insider trades:")
        for trade in trades[:5]:
            print(f"  {trade.to_plain_language()}")
    else:
        print("No trades found (may be rate limited or no recent filings)")

    print("\nDone.")
