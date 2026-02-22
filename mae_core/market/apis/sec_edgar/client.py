#!/usr/bin/env python3
"""SEC EDGAR API client with rate limiting."""

import re
import time
import html
import logging
import xml.etree.ElementTree as ET
from typing import List, Optional
import requests

from .models import Form8KEvent, InsiderTrade

logger = logging.getLogger(__name__)

# SEC EDGAR API endpoints
SEC_BASE_URL = "https://data.sec.gov"
SEC_WWW_URL = "https://www.sec.gov"
SEC_SUBMISSIONS_URL = f"{SEC_BASE_URL}/submissions"
SEC_TICKERS_URL = f"{SEC_WWW_URL}/files/company_tickers.json"

# User agent required by SEC (they block default requests user agents)
# SEC compliance: they block IPs with fake contact info
# Replace with real email before any live EDGAR queries
SEC_USER_AGENT = "MIDGE Trading Research midge@wardenclyffe.local"

# Rate limiting - SEC allows max 10 req/sec
REQUEST_DELAY = 0.15  # 150ms between requests


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
                        accession_number=accession_number
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
        """Parse Form 4 from HTML (XSLT-rendered) format."""
        trades = []

        try:
            filer_match = re.search(r'Reporting Person.*?<a[^>]*>([^<]+)</a>', html_content, re.DOTALL | re.IGNORECASE)
            filer_name = filer_match.group(1).strip() if filer_match else "Unknown"

            company_match = re.search(r'Issuer Name.*?class="FormData[^"]*"[^>]*>([^<]+)', html_content, re.DOTALL | re.IGNORECASE)
            company_name = company_match.group(1).strip() if company_match else "Unknown"

            ticker_match = re.search(r'Trading Symbol.*?class="FormData[^"]*"[^>]*>([^<]+)', html_content, re.DOTALL | re.IGNORECASE)
            ticker_symbol = ticker_match.group(1).strip() if ticker_match else ""

            relationships = []
            if re.search(r'Director.*?X|Director.*?checked', html_content, re.IGNORECASE):
                relationships.append("Director")
            if re.search(r'Officer.*?X|Officer.*?checked', html_content, re.IGNORECASE):
                relationships.append("Officer")
            if re.search(r'10%.*?Owner.*?X|10%.*?Owner.*?checked', html_content, re.IGNORECASE):
                relationships.append("10% Owner")

            filer_relationship = ", ".join(relationships) if relationships else "Other"

            title_match = re.search(r'Officer.*?Title.*?class="FormData[^"]*"[^>]*>([^<]+)', html_content, re.DOTALL | re.IGNORECASE)
            filer_title = title_match.group(1).strip() if title_match else ""

            table1_match = re.search(r'Table I.*?</tbody>\s*</table>', html_content, re.DOTALL | re.IGNORECASE)
            if table1_match:
                table_content = table1_match.group(0)

                price_pattern = re.compile(r'\$\s*</span>\s*<span[^>]*>([\d,]+\.?\d*)</span>', re.IGNORECASE)

                row_pattern = re.compile(r'<tr>\s*(.*?)\s*</tr>', re.DOTALL)
                rows = row_pattern.findall(table_content)

                for row in rows:
                    if '<th' in row:
                        continue

                    cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)

                    if len(cells) >= 8:
                        try:
                            security_title = re.sub(r'<[^>]+>', '', cells[0]).strip()

                            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', cells[1])
                            if not date_match:
                                continue
                            trans_date = date_match.group(1)

                            code_match = re.search(r'>([PSMDGFC])<', cells[3])
                            if not code_match:
                                continue
                            trans_code = code_match.group(1)

                            shares_match = re.search(r'>([\d,]+\.?\d*)<', cells[5])
                            if not shares_match:
                                continue
                            shares = float(shares_match.group(1).replace(',', ''))

                            ad_match = re.search(r'>([AD])<', cells[6])
                            if not ad_match:
                                continue
                            acq_disp = ad_match.group(1)

                            price_match = price_pattern.search(cells[7])
                            if not price_match:
                                price_match = re.search(r'>([\d,]+\.?\d*)<', cells[7])

                            if not price_match:
                                continue

                            price = float(price_match.group(1).replace(',', ''))

                            owned_match = re.search(r'>([\d,]+\.?\d*)<', cells[8])
                            shares_owned = float(owned_match.group(1).replace(',', '')) if owned_match else 0

                            if price > 0 and shares > 0:
                                trades.append(InsiderTrade(
                                    filer_name=filer_name,
                                    filer_title=filer_title,
                                    filer_relationship=filer_relationship,
                                    company_name=company_name,
                                    company_cik=cik,
                                    ticker_symbol=ticker_symbol,
                                    transaction_date=trans_date,
                                    transaction_type=acq_disp,
                                    shares=shares,
                                    price_per_share=price,
                                    total_value=shares * price,
                                    shares_owned_after=shares_owned,
                                    filing_date="",
                                    accession_number=accession_number,
                                    form_type="4",
                                    transaction_code=trans_code
                                ))
                        except (ValueError, IndexError):
                            continue

        except Exception as e:
            logger.error(f"Error parsing Form 4 HTML {accession_number}: {e}")

        return trades

    def _parse_transaction(self, trans_elem, **metadata) -> Optional[InsiderTrade]:
        """Parse a single transaction element from Form 4."""
        try:
            date_elem = trans_elem.find(".//transactionDate/value")
            trans_date = date_elem.text if date_elem is not None else None

            amounts = trans_elem.find("transactionAmounts")
            if amounts is None:
                return None

            shares_elem = amounts.find("transactionShares/value")
            shares = float(shares_elem.text) if shares_elem is not None and shares_elem.text else 0

            price_elem = amounts.find("transactionPricePerShare/value")
            price = float(price_elem.text) if price_elem is not None and price_elem.text else 0

            code_elem = amounts.find("transactionAcquiredDisposedCode/value")
            trans_type = code_elem.text if code_elem is not None else "A"

            holdings = trans_elem.find("postTransactionAmounts")
            shares_after = 0
            if holdings is not None:
                after_elem = holdings.find("sharesOwnedFollowingTransaction/value")
                if after_elem is not None and after_elem.text:
                    shares_after = float(after_elem.text)

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
                filing_date="",
                accession_number=metadata["accession_number"],
                form_type="4"
            )

        except Exception as e:
            logger.error(f"Error parsing transaction: {e}")
            return None

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
