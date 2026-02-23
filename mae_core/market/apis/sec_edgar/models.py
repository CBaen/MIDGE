#!/usr/bin/env python3
"""SEC EDGAR data models - Form8KEvent and InsiderTrade."""

from dataclasses import dataclass


@dataclass
class Form8KEvent:
    """Material event from SEC Form 8-K filing.

    Form 8-K must be filed within 4 business days of material events.
    These events often PRECEDE insider trading activity.
    """
    company_name: str
    company_cik: str
    ticker_symbol: str
    item_code: str
    item_description: str
    event_date: str
    event_summary: str
    filing_date: str
    accession_number: str
    document_url: str = ""
    form_type: str = "8-K"
    material_impact: str = "unknown"
    confidence: float = 0.70
    signal_source: str = "sec_form8k"
    decay_rate: float = 0.25  # ~3 day half-life (market prices binary 8-K events fast)

    ITEM_CODES = {
        "1.01": ("Material Agreement", "bullish"),
        "1.02": ("Termination of Agreement", "bearish"),
        "1.03": ("Bankruptcy", "bearish"),
        "2.01": ("Acquisition/Disposition", "neutral"),
        "2.02": ("Results of Operations", "neutral"),
        "2.03": ("Asset-Backed Securities", "neutral"),
        "2.04": ("Triggering Events", "bearish"),
        "2.05": ("Exit/Disposal", "bearish"),
        "2.06": ("Material Impairment", "bearish"),
        "3.01": ("Delisting", "bearish"),
        "3.02": ("Unregistered Sales", "neutral"),
        "3.03": ("Material Modification", "neutral"),
        "4.01": ("Auditor Changes", "neutral"),
        "4.02": ("Non-Reliance on Financials", "bearish"),
        "5.01": ("Change in Control", "neutral"),
        "5.02": ("Officer/Director Changes", "neutral"),
        "5.03": ("Bylaw Amendments", "neutral"),
        "5.04": ("Shareholder Waiver", "neutral"),
        "5.05": ("Compensatory Arrangements", "neutral"),
        "5.06": ("Shell Company Status", "neutral"),
        "5.07": ("Shareholder Voting", "neutral"),
        "5.08": ("Shareholder Nominations", "neutral"),
        "7.01": ("Regulation FD Disclosure", "neutral"),
        "8.01": ("Other Events", "neutral"),
        "9.01": ("Financial Statements/Exhibits", "neutral"),
    }

    def to_plain_language(self) -> str:
        impact_emoji = {
            "bullish": "+",
            "bearish": "-",
            "neutral": "~",
            "unknown": "?"
        }
        emoji = impact_emoji.get(self.material_impact, "?")
        return (
            f"[{emoji}] {self.company_name} ({self.ticker_symbol}): "
            f"{self.item_description} - {self.event_summary[:80]}"
        )

    @classmethod
    def get_item_info(cls, item_code: str) -> tuple:
        return cls.ITEM_CODES.get(item_code, ("Unknown Item", "unknown"))


@dataclass
class InsiderTrade:
    """Single insider trading transaction from Form 4."""
    filer_name: str
    filer_title: str
    filer_relationship: str
    company_name: str
    company_cik: str
    ticker_symbol: str
    transaction_date: str
    transaction_type: str
    transaction_code: str = ""
    shares: float = 0.0
    price_per_share: float = 0.0
    total_value: float = 0.0
    shares_owned_after: float = 0.0
    filing_date: str = ""
    accession_number: str = ""
    form_type: str = "4"
    is_plan_sale: bool = False  # True if 10b5-1 plan detected via footnotes
    footnotes: str = ""  # Raw footnote text from filing (for audit trail)
    signal_source: str = "insider"
    decay_rate: float = 0.035  # ~20 day half-life (Lakonishok & Lee 2001)

    @property
    def is_purchase(self) -> bool:
        return self.transaction_type in ("A", "P", "buy", "purchase")

    def to_plain_language(self) -> str:
        action = "bought" if self.transaction_type == "A" else "sold"
        return (
            f"{self.filer_name} ({self.filer_title}) {action} "
            f"${self.total_value:,.0f} of {self.ticker_symbol or self.company_name} "
            f"({self.shares:,.0f} shares @ ${self.price_per_share:.2f})"
        )
