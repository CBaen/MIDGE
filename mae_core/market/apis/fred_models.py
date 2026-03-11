"""fred_models.py - FRED data models and direction classification logic.

Extracted from fred_client.py. Contains:
  - FRED_SERIES: series ID -> (name, signal_type) mapping
  - MacroIndicator: dataclass for a single FRED observation
  - _determine_direction: classify a macro value as bullish/bearish/neutral
"""

from __future__ import annotations

from dataclasses import dataclass


# Series definitions: id -> (human_readable_name, signal_type)
FRED_SERIES: dict[str, tuple] = {
    "T10Y2Y":       ("10Y-2Y Treasury Spread (Yield Curve)", "yield_curve"),
    "BAMLH0A0HYM2": ("ICE BofA High Yield Spread", "credit_spread"),
    "VIXCLS":       ("VIX Volatility Index", "volatility"),
    "DFF":          ("Federal Funds Rate (Effective)", "rates"),
    "UNRATE":       ("Unemployment Rate", "employment"),
    "CPIAUCSL":     ("Consumer Price Index (All Urban)", "inflation"),
    # Forex-critical additions
    "DGS2":         ("2-Year Treasury Constant Maturity Rate", "treasury_2y"),
    "DGS10":        ("10-Year Treasury Constant Maturity Rate", "treasury_10y"),
    "T10Y3M":       ("10Y-3M Treasury Spread (Alt Recession Indicator)", "yield_curve_3m"),
    "DTWEXBGS":     ("US Dollar Index - Broad, Goods (DXY proxy)", "dollar_index"),
    "DBDI":         ("Baltic Dry Index (Shipping Demand Proxy)", "logistics"),
}


def _determine_direction(series_id: str, value: float) -> str:
    """Classify a macro value as bullish, bearish, or neutral.

    Rules are based on historically significant thresholds — not precise
    predictions, but reliable regime signals.

    Args:
        series_id: FRED series identifier
        value: Numeric value of the observation

    Returns:
        "bullish", "bearish", or "neutral"
    """
    if series_id == "T10Y2Y":
        # Yield curve inversion (negative spread) = recession signal = bearish
        # Healthy spread above 0.5 = bullish
        if value < 0:
            return "bearish"
        elif value > 0.5:
            return "bullish"
        return "neutral"

    elif series_id == "BAMLH0A0HYM2":
        # Credit spread: high = credit stress = fear = bearish
        # Low spread = risk-on, credit healthy = bullish
        if value > 5.0:
            return "bearish"
        elif value < 3.0:
            return "bullish"
        return "neutral"

    elif series_id == "VIXCLS":
        # VIX: high = fear = bearish. Low = complacency/calm = bullish
        if value > 30:
            return "bearish"
        elif value < 15:
            return "bullish"
        return "neutral"

    elif series_id == "DFF":
        # Rates: just reporting — direction is context-dependent
        # High rates = tighter conditions but not directly bullish/bearish
        return "neutral"

    elif series_id in ("DGS2", "DGS10"):
        # Raw treasury yields: rising = bearish for equities/bonds, bullish for USD
        # Report neutral — used for spread computation context
        return "neutral"

    elif series_id == "T10Y3M":
        # Alt recession indicator: 10Y minus 3-month T-bill
        # Inversion (negative) more reliable recession predictor than T10Y2Y
        if value < 0:
            return "bearish"
        elif value > 1.0:
            return "bullish"
        return "neutral"

    elif series_id == "DTWEXBGS":
        # Broad Dollar Index (DXY proxy): rising dollar = bearish for commodities/EM
        # Thresholds: historically around 100-130 range for the broad index
        # Use change-based direction via a level band
        if value > 120:
            return "bearish"   # Very strong dollar = bearish risk assets
        elif value < 105:
            return "bullish"   # Weak dollar = bullish commodities, EM, risk
        return "neutral"

    elif series_id == "UNRATE":
        # Unemployment: high = economic weakness = bearish
        # Low = tight labor market = bullish
        if value > 5.0:
            return "bearish"
        elif value < 4.0:
            return "bullish"
        return "neutral"

    elif series_id == "CPIAUCSL":
        # CPI level: raw level is not directly directional
        # Month-over-month change would be more useful, but keep simple
        return "neutral"

    return "neutral"


@dataclass
class MacroIndicator:
    """A single macroeconomic indicator reading from FRED.

    Carries the same signal metadata fields as other MIDGE signals
    (signal_source, decay_rate, confidence) so the convergence alerter
    can weight macro context against micro signals uniformly.

    Macro signals decay slowly (~70 day half-life via decay_rate=0.01)
    because economic regimes persist for months, not days.
    """

    series_id: str          # e.g. "T10Y2Y"
    series_name: str        # e.g. "10Y-2Y Treasury Spread (Yield Curve)"
    value: float            # Numeric observation value
    date: str               # YYYY-MM-DD of the observation
    signal_type: str        # "yield_curve", "credit_spread", "volatility", "rates", "employment", "inflation"
    direction: str          # "bullish", "bearish", or "neutral"
    signal_source: str = "fred_macro"
    decay_rate: float = 0.01    # Slow decay — macro regimes persist ~70 days
    confidence: float = 0.70    # Government data is reliable but lags reality
