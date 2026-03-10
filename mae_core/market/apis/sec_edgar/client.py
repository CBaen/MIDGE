#!/usr/bin/env python3
"""SEC EDGAR API client with rate limiting.

Sub-modules:
  sec_edgar_parsers.py — _parse_form4_html, _parse_transaction
"""

import re
import time
import html
import logging
import xml.etree.ElementTree as ET
from typing import List, Optional
import requests

from .models import Form8KEvent, InsiderTrade
from .sec_edgar_parsers import _parse_form4_html as _parse_form4_html_fn
from .sec_edgar_parsers import _parse_transaction as _parse_transaction_fn

logger = logging.getLogger(__name__)

# SEC EDGAR API endpoints
SEC_BASE_URL = "https://data.sec.gov"
SEC_WWW_URL = "https://www.sec.gov"
SEC_SUBMISSIONS_URL = f"{SEC_BASE_URL}/submissions"
SEC_TICKERS_URL = f"{SEC_WWW_URL}/files/company_tickers.json"

# User agent required by SEC (they block default requests user agents)
# SEC compliance: they block IPs with fake contact info
SEC_USER_AGENT = "MIDGE Trading Research cameronbpaul@gmail.com"

# Rate limiting - SEC allows max 10 req/sec
REQUEST_DELAY = 0.15  # 150ms between requests


class _ResponseShim:
    """Minimal Response-like wrapper for MarketDataProvider results.

    Allows SEC EDGAR callers to continue using .json(), .text, .content
    without knowing the request was routed through the provider.
    """

    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    @property
    def text(self) -> str:
        if isinstance(self._payload, dict) and "text" in self._payload:
            return self._payload["text"]
        return ""

    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8")

    def json(self):
        return self._payload


class SECEdgarClient:
    """
    Client for SEC EDGAR API.

    Respects rate limits and uses required User-Agent header.
    """

    def __init__(self, user_agent: str = SEC_USER_AGENT, provider=None, raw_store=None):
        self._provider = provider
        self._raw_store = raw_store
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

        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            resp = market_request(
                self._provider, url,
                headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
                params=params, source_name="sec_edgar", timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return _ResponseShim(resp.payload)
            logger.warning("SEC EDGAR error via provider: %s", resp.error_message)
            return None

        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response
            else:
                logger.warning(f"SEC EDGAR error: {response.status_code} for {url}")
                return None
        except Exception as e:
            logger.error(f"SEC EDGAR request failed: {e}")
            return None

    def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK number for a ticker symbol.

        CIK is the Central Index Key - SEC's company identifier.
        """
        response = self._get(SEC_TICKERS_URL)

        if not response:
            return None

        try:
            data = response.json()
            ticker_upper = ticker.upper()

            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker_upper:
                    cik = str(entry.get("cik_str", ""))
                    return cik.zfill(10)

            return None
        except Exception as e:
            logger.error(f"Error parsing CIK mapping: {e}")
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
        submissions_url = f"{SEC_SUBMISSIONS_URL}/CIK{cik}.json"
        response = self._get(submissions_url)

        if not response:
            return []

        try:
            data = response.json()
            filings = data.get("filings", {}).get("recent", {})

            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            primary_docs = filings.get("primaryDocument", [])

            results = []
            for i, form in enumerate(forms):
                if form == form_type:
                    acc_no_dash = accessions[i].replace("-", "") if i < len(accessions) else ""
                    primary_doc = primary_docs[i] if i < len(primary_docs) else ""
                    cik_int = int(cik)

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
            logger.error(f"Error parsing filings: {e}")
            return []

    def parse_form4(self, cik: str, accession_number: str, document_url: str = None) -> List[InsiderTrade]:
        """
        Parse a Form 4 filing to extract insider trades.

        Args:
            cik: Company CIK
            accession_number: Filing accession number
            document_url: Direct URL to the document (preferred)

        Returns:
            List of InsiderTrade objects
        """
        response = None

        if document_url:
            filename_match = re.search(r'/([^/]+\.xml)$', document_url)
            if filename_match:
                filename = filename_match.group(1)
                if '/xslF345X05/' not in document_url:
                    xslt_url = document_url.replace(f'/{filename}', f'/xslF345X05/{filename}')
                    response = self._get(xslt_url)

                    if not response or (response and '<html' not in response.text.lower()):
                        response = self._get(document_url)
                else:
                    response = self._get(document_url)
            else:
                response = self._get(document_url)
        else:
            acc_clean = accession_number.replace("-", "")
            cik_int = int(cik)

            for filename in ['doc4.xml', 'form4.xml']:
                xslt_url = f"{SEC_WWW_URL}/Archives/edgar/data/{cik_int}/{acc_clean}/xslF345X05/{filename}"
                response = self._get(xslt_url)

                if response and '<html' in response.text.lower():
                    break

            if not response:
                xml_url = f"{SEC_WWW_URL}/Archives/edgar/data/{cik_int}/{acc_clean}/{accession_number}.xml"
                response = self._get(xml_url)

        if not response:
            return []

        content = response.text

        if content.strip().startswith("<!DOCTYPE html") or "<html" in content[:500]:
            return self._parse_form4_html(content, cik, accession_number)

        try:
            root = ET.fromstring(response.content)

            reporting_owner = root.find(".//reportingOwner") or root.find("reportingOwner")

            filer_name = ""
            filer_title = ""
            filer_relationship = []

            if reporting_owner is not None:
                owner_id = reporting_owner.find("reportingOwnerId")
                if owner_id is not None:
                    name_elem = owner_id.find("rptOwnerName")
                    if name_elem is not None:
                        filer_name = name_elem.text or ""

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
                            title_elem = relationship.find("officerTitle")
                            if title_elem is not None:
                                filer_title = title_elem.text or ""

                    if relationship.find("isTenPercentOwner") is not None:
                        is_ten = relationship.find("isTenPercentOwner").text
                        if is_ten and is_ten.lower() in ("1", "true"):
                            filer_relationship.append("10% Owner")

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

            # Collect footnotes — 10b5-1 plan sales are disclosed here
            footnotes_text = ""
            is_plan_sale = False
            footnotes_elem = root.find(".//footnotes") or root.find("footnotes")
            if footnotes_elem is not None:
                parts = []
                for fn in footnotes_elem:
                    text = fn.text or ""
                    if text:
                        parts.append(text)
                footnotes_text = " ".join(parts)
                # Detect 10b5-1 plan references
                fn_lower = footnotes_text.lower()
                if "10b5-1" in fn_lower or "10b-5" in fn_lower or "rule 10b" in fn_lower:
                    is_plan_sale = True

            trades = []

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
                        accession_number=accession_number,
                        is_plan_sale=is_plan_sale,
                        footnotes=footnotes_text,
                    )
                    if trade:
                        trades.append(trade)

            return trades

        except ET.ParseError as e:
            logger.error(f"XML parse error for {accession_number}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing Form 4 {accession_number}: {e}")
            return []

    def _parse_form4_html(self, html_content: str, cik: str, accession_number: str) -> List[InsiderTrade]:
        """Parse Form 4 from HTML (XSLT-rendered) format.

        Delegates to sec_edgar_parsers._parse_form4_html.
        """
        return _parse_form4_html_fn(html_content, cik, accession_number)

    def _parse_transaction(self, trans_elem, **metadata) -> Optional[InsiderTrade]:
        """Parse a single transaction element from Form 4.

        Delegates to sec_edgar_parsers._parse_transaction.
        """
        return _parse_transaction_fn(trans_elem, **metadata)

    def parse_form8k(self, cik: str, accession_number: str, filing_date: str,
                     company_name: str, ticker_symbol: str, document_url: str = None) -> List[Form8KEvent]:
        """
        Parse a Form 8-K filing to extract material events.

        Args:
            cik: Company CIK
            accession_number: Filing accession number
            filing_date: Date of filing
            company_name: Company name
            ticker_symbol: Stock ticker
            document_url: Direct URL to the document

        Returns:
            List of Form8KEvent objects
        """
        if document_url and document_url.endswith(('.htm', '.html', '.txt')):
            doc_url = document_url
        else:
            acc_clean = accession_number.replace("-", "")
            cik_int = int(cik)
            doc_url = f"{SEC_WWW_URL}/Archives/edgar/data/{cik_int}/{acc_clean}"

        response = self._get(doc_url)

        if not response:
            return []

        try:
            doc_content = response.text
            doc_link = doc_url

            events = []

            item_pattern = r'(?:Item|ITEM)\s+(\d+\.\d+)'
            items_found = re.findall(item_pattern, doc_content, re.IGNORECASE)

            seen = set()
            unique_items = []
            for item in items_found:
                if item not in seen:
                    seen.add(item)
                    unique_items.append(item)

            for item_code in unique_items:
                description, impact = Form8KEvent.get_item_info(item_code)

                context_pattern = rf'(?:Item|ITEM)\s+{re.escape(item_code)}[:\s\-]*([^<\n]{{0,200}})'
                context_match = re.search(context_pattern, doc_content, re.IGNORECASE)
                summary = context_match.group(1).strip() if context_match else description

                summary = html.unescape(summary)
                summary = re.sub(r'\s+', ' ', summary).strip()
                if len(summary) < 10:
                    summary = description

                event = Form8KEvent(
                    company_name=company_name,
                    company_cik=cik,
                    ticker_symbol=ticker_symbol,
                    item_code=item_code,
                    item_description=description,
                    event_date=filing_date,
                    event_summary=summary[:200],
                    filing_date=filing_date,
                    accession_number=accession_number,
                    document_url=doc_link,
                    material_impact=impact,
                    confidence=0.70 if impact != "unknown" else 0.50
                )
                events.append(event)

            return events

        except Exception as e:
            logger.error(f"Error parsing Form 8-K: {e}")
            return []
