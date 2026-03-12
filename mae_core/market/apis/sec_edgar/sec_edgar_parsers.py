"""sec_edgar_parsers.py - Form 4 HTML and XML transaction parsing helpers.

Extracted from sec_edgar/client.py. Contains:
  - _parse_form4_html: Parse Form 4 from XSLT-rendered HTML format
  - _parse_transaction: Parse a single nonDerivativeTransaction XML element
  - _parse_derivative_transaction: Parse a single derivativeTransaction XML element (Table II)
"""

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from mae_core.market.apis.sec_edgar.models import InsiderTrade

logger = logging.getLogger(__name__)


def _parse_form4_html(
    html_content: str,
    cik: str,
    accession_number: str,
) -> "List[InsiderTrade]":
    """Parse Form 4 from HTML (XSLT-rendered) format.

    Called when the SEC EDGAR response is HTML rather than raw XML —
    the XSLT stylesheet renders Form 4 as a human-readable HTML table.
    Extracts Table I (non-derivative transactions) rows.
    """
    from mae_core.market.apis.sec_edgar.models import InsiderTrade

    trades = []

    try:
        # Detect 10b5-1 plan references in footnotes section of HTML
        html_lower = html_content.lower()
        is_plan_sale = (
            "10b5-1" in html_lower
            or "10b-5" in html_lower
            or "rule 10b" in html_lower
        )
        # Extract footnote text for audit trail
        footnotes_text = ""
        fn_matches = re.findall(
            r'class="Footnote[^"]*"[^>]*>(.*?)</(?:td|span|div)',
            html_content,
            re.DOTALL | re.IGNORECASE,
        )
        if fn_matches:
            footnotes_text = " ".join(
                re.sub(r"<[^>]+>", "", m).strip() for m in fn_matches
            )

        filer_match = re.search(
            r"Reporting Person.*?<a[^>]*>([^<]+)</a>",
            html_content,
            re.DOTALL | re.IGNORECASE,
        )
        filer_name = filer_match.group(1).strip() if filer_match else "Unknown"

        company_match = re.search(
            r'Issuer Name.*?class="FormData[^"]*"[^>]*>([^<]+)',
            html_content,
            re.DOTALL | re.IGNORECASE,
        )
        company_name = company_match.group(1).strip() if company_match else "Unknown"

        ticker_match = re.search(
            r'Trading Symbol.*?class="FormData[^"]*"[^>]*>([^<]+)',
            html_content,
            re.DOTALL | re.IGNORECASE,
        )
        ticker_symbol = ticker_match.group(1).strip() if ticker_match else ""

        relationships = []
        if re.search(r"Director.*?X|Director.*?checked", html_content, re.IGNORECASE):
            relationships.append("Director")
        if re.search(r"Officer.*?X|Officer.*?checked", html_content, re.IGNORECASE):
            relationships.append("Officer")
        if re.search(
            r"10%.*?Owner.*?X|10%.*?Owner.*?checked", html_content, re.IGNORECASE
        ):
            relationships.append("10% Owner")

        filer_relationship = ", ".join(relationships) if relationships else "Other"

        title_match = re.search(
            r'Officer.*?Title.*?class="FormData[^"]*"[^>]*>([^<]+)',
            html_content,
            re.DOTALL | re.IGNORECASE,
        )
        filer_title = title_match.group(1).strip() if title_match else ""

        table1_match = re.search(
            r"Table I.*?</tbody>\s*</table>",
            html_content,
            re.DOTALL | re.IGNORECASE,
        )
        if table1_match:
            table_content = table1_match.group(0)

            price_pattern = re.compile(
                r'\$\s*</span>\s*<span[^>]*>([\d,]+\.?\d*)</span>',
                re.IGNORECASE,
            )

            row_pattern = re.compile(r"<tr>\s*(.*?)\s*</tr>", re.DOTALL)
            rows = row_pattern.findall(table_content)

            for row in rows:
                if "<th" in row:
                    continue

                cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)

                if len(cells) >= 8:
                    try:
                        security_title = re.sub(r"<[^>]+>", "", cells[0]).strip()

                        date_match = re.search(r"(\d{2}/\d{2}/\d{4})", cells[1])
                        if not date_match:
                            continue
                        trans_date = date_match.group(1)

                        code_match = re.search(r">([PSMDGFC])<", cells[3])
                        if not code_match:
                            continue
                        trans_code = code_match.group(1)

                        shares_match = re.search(r">([\d,]+\.?\d*)<", cells[5])
                        if not shares_match:
                            continue
                        shares = float(shares_match.group(1).replace(",", ""))

                        ad_match = re.search(r">([AD])<", cells[6])
                        if not ad_match:
                            continue
                        acq_disp = ad_match.group(1)

                        price_match = price_pattern.search(cells[7])
                        if not price_match:
                            price_match = re.search(r">([\d,]+\.?\d*)<", cells[7])

                        if not price_match:
                            continue

                        price = float(price_match.group(1).replace(",", ""))

                        owned_match = re.search(r">([\d,]+\.?\d*)<", cells[8])
                        shares_owned = (
                            float(owned_match.group(1).replace(",", ""))
                            if owned_match
                            else 0
                        )

                        if price > 0 and shares > 0:
                            trades.append(
                                InsiderTrade(
                                    filer_name=filer_name,
                                    filer_title=filer_title,
                                    filer_relationship=filer_relationship,
                                    company_name=company_name,
                                    company_cik=cik,
                                    ticker_symbol=ticker_symbol,
                                    transaction_date=trans_date,
                                    transaction_type=acq_disp,
                                    transaction_code=trans_code,
                                    shares=shares,
                                    price_per_share=price,
                                    total_value=shares * price,
                                    shares_owned_after=shares_owned,
                                    filing_date="",
                                    accession_number=accession_number,
                                    form_type="4",
                                    is_plan_sale=is_plan_sale,
                                    footnotes=footnotes_text,
                                )
                            )
                    except (ValueError, IndexError):
                        continue

    except Exception as e:
        logger.error(f"Error parsing Form 4 HTML {accession_number}: {e}")

    return trades


def _parse_transaction(
    trans_elem: ET.Element,
    **metadata,
) -> "Optional[InsiderTrade]":
    """Parse a single nonDerivativeTransaction element from Form 4 XML."""
    from mae_core.market.apis.sec_edgar.models import InsiderTrade

    try:
        date_elem = trans_elem.find(".//transactionDate/value")
        trans_date = date_elem.text if date_elem is not None else None

        # Extract transaction code (S=sale, P=purchase, M=option exercise, etc.)
        coding = trans_elem.find("transactionCoding")
        trans_code = ""
        if coding is not None:
            code_elem = coding.find("transactionCode")
            if code_elem is not None and code_elem.text:
                trans_code = code_elem.text.strip()

        amounts = trans_elem.find("transactionAmounts")
        if amounts is None:
            return None

        shares_elem = amounts.find("transactionShares/value")
        shares = (
            float(shares_elem.text)
            if shares_elem is not None and shares_elem.text
            else 0
        )

        price_elem = amounts.find("transactionPricePerShare/value")
        price = (
            float(price_elem.text)
            if price_elem is not None and price_elem.text
            else 0
        )

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
            transaction_code=trans_code,
            shares=shares,
            price_per_share=price,
            total_value=shares * price,
            shares_owned_after=shares_after,
            filing_date="",
            accession_number=metadata["accession_number"],
            form_type="4",
            is_plan_sale=metadata.get("is_plan_sale", False),
            footnotes=metadata.get("footnotes", ""),
        )

    except Exception as e:
        logger.error(f"Error parsing transaction: {e}")
        return None


def _parse_derivative_transaction(
    dtrans_elem: ET.Element,
    is_plan_sale: bool = False,
) -> "Optional[DerivativeTransaction]":
    """Parse a single derivativeTransaction element from Form 4 Table II (XML).

    Covers options exercises (code M), warrant conversions (code C), RSU
    vestings, and other derivative security transactions.

    Options exercises near earnings where the insider HOLDS (doesn't sell on
    the same day via a paired disposition) are a documented bullish signal.
    """
    from mae_core.market.apis.sec_edgar.models import DerivativeTransaction

    try:
        # Security title (e.g. "Employee Stock Option (right to buy)")
        title_elem = dtrans_elem.find(".//securityTitle/value")
        security_title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

        # Transaction date
        date_elem = dtrans_elem.find(".//transactionDate/value")
        transaction_date = date_elem.text if date_elem is not None and date_elem.text else ""

        # Transaction code (M=option exercise, C=conversion, S=sale, etc.)
        coding = dtrans_elem.find("transactionCoding")
        trans_code = ""
        if coding is not None:
            code_elem = coding.find("transactionCode")
            if code_elem is not None and code_elem.text:
                trans_code = code_elem.text.strip()

        # Transaction amounts
        amounts = dtrans_elem.find("transactionAmounts")
        if amounts is None:
            return None

        shares_elem = amounts.find("transactionShares/value")
        shares = float(shares_elem.text) if shares_elem is not None and shares_elem.text else 0.0

        price_elem = amounts.find("transactionPricePerShare/value")
        price_per_share = (
            float(price_elem.text) if price_elem is not None and price_elem.text else 0.0
        )

        acq_disp_elem = amounts.find("transactionAcquiredDisposedCode/value")
        trans_type = acq_disp_elem.text if acq_disp_elem is not None and acq_disp_elem.text else "A"

        # Exercise price (column 3B in Form 4) — the option strike price
        exercise_elem = dtrans_elem.find(".//conversionOrExercisePrice/value")
        exercise_price = (
            float(exercise_elem.text) if exercise_elem is not None and exercise_elem.text else 0.0
        )

        # Expiration date (column 3C)
        expiry_elem = dtrans_elem.find(".//expirationDate/value")
        expiration_date = (
            expiry_elem.text.strip() if expiry_elem is not None and expiry_elem.text else ""
        )

        # Underlying security shares (for options, the number of common shares deliverable)
        underlying_elem = dtrans_elem.find(".//underlyingSecurityShares/value")
        underlying_shares = (
            float(underlying_elem.text)
            if underlying_elem is not None and underlying_elem.text
            else 0.0
        )

        # Shares owned after (post-transaction holdings of the derivative security)
        post_elem = dtrans_elem.find(".//sharesOwnedFollowingTransaction/value")
        if post_elem is None:
            post_elem = dtrans_elem.find(".//postTransactionAmounts/sharesOwnedFollowingTransaction/value")
        shares_owned_after = (
            float(post_elem.text) if post_elem is not None and post_elem.text else 0.0
        )

        # Require at least a transaction code or shares to consider this valid
        if not trans_code and shares == 0.0:
            return None

        return DerivativeTransaction(
            security_title=security_title,
            transaction_date=transaction_date,
            transaction_code=trans_code,
            transaction_type=trans_type,
            shares=shares,
            price_per_share=price_per_share,
            exercise_price=exercise_price,
            expiration_date=expiration_date,
            underlying_shares=underlying_shares,
            shares_owned_after=shares_owned_after,
            is_plan_sale=is_plan_sale,
        )

    except Exception as e:
        logger.error(f"Error parsing derivative transaction: {e}")
        return None
