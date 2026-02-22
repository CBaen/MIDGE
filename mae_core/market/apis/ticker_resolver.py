#!/usr/bin/env python3
"""
ticker_resolver.py - Map company names to ticker symbols

Government data sources (USASpending, SAM.gov) return company legal names
like "LOCKHEED MARTIN CORPORATION" or "RAYTHEON TECHNOLOGIES CORP".
This service resolves those to tradeable ticker symbols.

Strategy:
1. Curated mapping of major government contractors (high-confidence)
2. Alias expansion for common name variations
3. Fuzzy matching for close-but-not-exact names

Designed to be called by ContractPredictor, ConvergenceAlerter, and
any module that receives a company name from government data.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"

# Suffixes to strip for matching
_CORP_SUFFIXES = re.compile(
    r"\s*\b(CORPORATION|CORP|INCORPORATED|INC|LLC|LTD|LIMITED|LP|"
    r"L\.P\.|CO|COMPANY|GROUP|HOLDINGS|HOLDING|INTL|INTERNATIONAL|"
    r"TECHNOLOGIES|TECHNOLOGY|TECH|SYSTEMS|SERVICES|SOLUTIONS|"
    r"USA|US|AMERICA|AMERICAS|NA|NORTH AMERICA)\b\.?\s*",
    re.IGNORECASE,
)


def _normalize(name: str) -> str:
    """Normalize a company name for matching."""
    upper = name.upper().strip()
    # Strip leading "THE"
    upper = re.sub(r"^THE\s+", "", upper)
    # Strip corporate suffixes
    cleaned = _CORP_SUFFIXES.sub(" ", upper)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ── Curated mapping: normalized name → ticker ──────────────────────────

_CURATED: Dict[str, str] = {
    # Defense / Aerospace
    "LOCKHEED MARTIN": "LMT",
    "RAYTHEON": "RTX",
    "RTX": "RTX",
    "NORTHROP GRUMMAN": "NOC",
    "GENERAL DYNAMICS": "GD",
    "BOEING": "BA",
    "L3HARRIS": "LHX",
    "L3 HARRIS": "LHX",
    "BAE": "BAESY",
    "LEIDOS": "LDOS",
    "SAIC": "SAIC",
    "BOOZ ALLEN HAMILTON": "BAH",
    "BOOZ ALLEN": "BAH",
    "HUNTINGTON INGALLS": "HII",
    "TEXTRON": "TXT",
    "KRATOS DEFENSE": "KTOS",
    "KRATOS": "KTOS",
    "CURTISS WRIGHT": "CW",
    "CURTISS-WRIGHT": "CW",
    "HOWMET AEROSPACE": "HWM",
    "HOWMET": "HWM",
    "MERCURY": "MRCY",
    "BWX": "BWXT",
    "TRANSDIGM": "TDG",
    "HEICO": "HEI",
    "AXON ENTERPRISE": "AXON",
    "AXON": "AXON",
    "PALANTIR": "PLTR",
    "ANDURIL": "",  # Private — empty string = known but not tradeable

    # Big Tech (government contracts)
    "AMAZON": "AMZN",
    "AMAZON WEB": "AMZN",
    "AWS": "AMZN",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "ORACLE": "ORCL",
    "IBM": "IBM",
    "GENERAL ELECTRIC": "GE",
    "GE AEROSPACE": "GE",
    "CISCO": "CSCO",
    "DELL": "DELL",
    "HEWLETT PACKARD ENTERPRISE": "HPE",
    "HP": "HPQ",
    "ACCENTURE": "ACN",

    # Health / Pharma (HHS contracts)
    "PFIZER": "PFE",
    "JOHNSON JOHNSON": "JNJ",
    "MODERNA": "MRNA",
    "MERCK": "MRK",
    "ABBVIE": "ABBV",
    "UNITEDHEALTH": "UNH",
    "MCKESSON": "MCK",
    "HUMANA": "HUM",
    "CIGNA": "CI",
    "ELEVANCE HEALTH": "ELV",
    "CENTENE": "CNC",

    # Infrastructure / Construction
    "FLUOR": "FLR",
    "JACOBS": "J",
    "AECOM": "ACM",
    "KBR": "KBR",
    "PARSONS": "PSN",
    "BECHTEL": "",  # Private

    # Energy
    "EXXON MOBIL": "XOM",
    "EXXONMOBIL": "XOM",
    "CHEVRON": "CVX",
    "CONOCOPHILLIPS": "COP",
    "HALLIBURTON": "HAL",
    "BAKER HUGHES": "BKR",
    "SCHLUMBERGER": "SLB",

    # Telecom / Comms
    "AT&T": "T",
    "ATT": "T",
    "VERIZON": "VZ",
    "T-MOBILE": "TMUS",
    "MOTOROLA": "MSI",

    # Vehicles / Logistics
    "GENERAL MOTORS": "GM",
    "FORD": "F",
    "OSHKOSH": "OSK",
    "CUMMINS": "CMI",
    "CATERPILLAR": "CAT",
    "DEERE": "DE",
    "JOHN DEERE": "DE",
    "FEDEX": "FDX",
    "UPS": "UPS",
    "UNITED PARCEL": "UPS",
}


def _load_extended_mapping() -> Dict[str, str]:
    """Load extended company→ticker mapping from data file if available."""
    path = _DATA_DIR / "company_tickers.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {_normalize(k): v for k, v in data.items()}
    except Exception:
        logger.warning("Failed to load extended ticker mapping", exc_info=True)
        return {}


# Build the full lookup on module load
_EXTENDED = _load_extended_mapping()
_FULL_MAP: Dict[str, str] = {**_CURATED, **_EXTENDED}


def resolve(company_name: str) -> Optional[str]:
    """
    Resolve a company name to a ticker symbol.

    Returns:
        Ticker string (e.g., "LMT"), empty string if known-private,
        or None if no match found.
    """
    normed = _normalize(company_name)

    # 1. Exact match on normalized name
    if normed in _FULL_MAP:
        return _FULL_MAP[normed] or None  # "" → None (private)

    # 2. Check if normalized name starts with any known key
    for key, ticker in _FULL_MAP.items():
        if normed.startswith(key) or key.startswith(normed):
            if ticker:
                return ticker

    # 3. Token overlap — at least 2 significant tokens match
    normed_tokens = set(normed.split())
    best_score = 0
    best_ticker = None
    for key, ticker in _FULL_MAP.items():
        if not ticker:
            continue
        key_tokens = set(key.split())
        overlap = len(normed_tokens & key_tokens)
        if overlap >= 2 and overlap > best_score:
            best_score = overlap
            best_ticker = ticker

    return best_ticker


def resolve_batch(company_names: List[str]) -> Dict[str, Optional[str]]:
    """Resolve multiple company names at once."""
    return {name: resolve(name) for name in company_names}


def get_known_tickers() -> Dict[str, str]:
    """Return a copy of the full curated mapping (for inspection/debugging)."""
    return dict(_FULL_MAP)
