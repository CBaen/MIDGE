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
    "TSIFRGHT":     ("Freight Transportation Services Index (Logistics Demand)", "logistics"),
    # Fed inflation targets — the Fed's actual policy anchor (not CPI)
    "PCEPI":        ("PCE Price Index (Fed Preferred Inflation Measure)", "inflation_pce"),
    "PCEPILFE":     ("Core PCE Price Index (Excludes Food & Energy)", "inflation_core_pce"),
    # Consumer demand
    "RSXFS":        ("Retail Sales Excluding Food Services", "consumer_spending"),
    "UMCSENT":      ("University of Michigan Consumer Sentiment", "consumer_sentiment"),
    # Money supply — liquidity condition for all risk assets
    "M2SL":         ("M2 Money Supply", "money_supply"),
    # Inflation expectations — market-implied, forward-looking
    "T5YIE":        ("5-Year Breakeven Inflation Rate", "inflation_expectations"),
    # Housing — rate-sensitive leading indicator
    "HOUST":        ("Housing Starts (Thousands of Units)", "housing"),
    "PERMIT":       ("Building Permits (Thousands of Units)", "housing_permits"),
    # Commodities via FRED — cross-asset convergence anchors
    "DCOILWTICO":   ("WTI Crude Oil Price (Dollars per Barrel)", "energy_price"),
    "GOLDAMGBD228NLBM": ("Gold Price London Fix (USD per Troy Oz)", "gold_price"),
    # Credit stress indicators
    "BAMLC0A0CM":   ("ICE BofA Investment Grade Corporate Bond Spread", "credit_spread_ig"),
    "TEDRATE":      ("TED Spread (3-Month LIBOR minus T-Bill, Bank Stress)", "bank_stress"),
    # Labor market — weekly pulse (highest-frequency FRED series)
    "ICSA":         ("Initial Jobless Claims (Seasonally Adjusted)", "jobless_claims"),
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

    elif series_id == "TSIFRGHT":
        # Freight Transportation Services Index: BTS monthly index (base ~100)
        # Rising = shipping demand expanding = bullish for industrials/commodities
        # Historical range: ~90 (2009 trough, 2020 COVID) to ~120+ (boom)
        if value > 115:
            return "bullish"
        elif value < 100:
            return "bearish"
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

    elif series_id == "PCEPI":
        # PCE Price Index (index level, base ~100 in 2012 dollars)
        # Raw level is not directional — used for month-over-month context
        return "neutral"

    elif series_id == "PCEPILFE":
        # Core PCE: the Fed's primary policy target. YoY above 2.5% = hawkish risk.
        # Raw index level: not directly actionable without delta, return neutral
        return "neutral"

    elif series_id == "RSXFS":
        # Retail sales (millions of dollars, seasonally adjusted)
        # High absolute level = strong consumer = bullish; low = contraction = bearish
        # Historically ~400K-600K+ range (in millions). Threshold is regime-relative.
        # Use percent-change context is ideal but level-based proxy here:
        return "neutral"

    elif series_id == "UMCSENT":
        # University of Michigan Consumer Sentiment (index, 1966=100 baseline)
        # Historical range: ~55 (troughs: 2008, 2022) to ~100+ (expansions)
        if value >= 85:
            return "bullish"   # Strong sentiment = consumer spending confidence
        elif value <= 65:
            return "bearish"   # Weak sentiment = consumers pulling back
        return "neutral"

    elif series_id == "M2SL":
        # M2 Money Supply (billions of dollars, seasonally adjusted)
        # Rapid expansion = liquidity tailwind for risk assets
        # Contraction (rare) = tightening conditions = bearish
        # Level alone doesn't signal direction; growth rate matters more
        return "neutral"

    elif series_id == "T5YIE":
        # 5-Year Breakeven Inflation Rate (market-implied, %)
        # Above Fed 2% target and rising = hawkish pressure = bearish for bonds/growth stocks
        # At or below 2% = inflation anchored = bullish for risk assets
        if value > 3.0:
            return "bearish"   # Inflation expectations dangerously unanchored
        elif value < 2.0:
            return "bullish"   # Inflation under control, Fed not forced to hike
        return "neutral"

    elif series_id == "HOUST":
        # Housing Starts (thousands of units, seasonally adjusted annual rate)
        # Historical range: ~500 (2009 trough) to ~1800 (2006 peak), ~1400-1600 normal
        if value >= 1400:
            return "bullish"   # Strong construction = economic confidence, labor demand
        elif value <= 900:
            return "bearish"   # Housing recession territory
        return "neutral"

    elif series_id == "PERMIT":
        # Building Permits (thousands of units, SAAR) — leading indicator for housing starts
        # Similar thresholds to HOUST — permits lead starts by ~1-3 months
        if value >= 1400:
            return "bullish"
        elif value <= 900:
            return "bearish"
        return "neutral"

    elif series_id == "DCOILWTICO":
        # WTI Crude Oil Price (dollars per barrel)
        # High oil = energy sector bullish, but also inflation/consumer squeeze = mixed
        # As a commodity signal: high = strong demand signal / supply crunch = bullish energy
        # Very high (>$90) = macro headwind for equities = bearish
        if value > 90:
            return "bearish"   # Demand-destroying / margin-compressing for non-energy
        elif value < 50:
            return "bullish"   # Input cost relief = consumer/industrial tailwind
        return "neutral"

    elif series_id == "GOLDAMGBD228NLBM":
        # Gold Price (USD per troy oz, London afternoon fix)
        # Gold rising = fear/inflation hedge demand = risk-off = bearish for equities
        # Gold falling = risk-on rotation away from safe haven = bullish for equities
        if value > 2500:
            return "bearish"   # Extreme safe-haven demand = systemic fear
        elif value < 1700:
            return "bullish"   # Low fear premium, risk assets preferred
        return "neutral"

    elif series_id == "BAMLC0A0CM":
        # ICE BofA Investment Grade Corporate Bond OAS (option-adjusted spread, %)
        # Low spread = healthy credit markets = bullish; high = credit stress = bearish
        # Historical range: ~0.5% (risk-on) to ~3%+ (crisis)
        if value > 2.0:
            return "bearish"   # Investment grade credit under stress
        elif value < 1.0:
            return "bullish"   # Tight spreads = healthy corporate credit
        return "neutral"

    elif series_id == "TEDRATE":
        # TED Spread (3-month LIBOR minus 3-month T-bill rate, %)
        # Measures bank-to-bank lending stress. High = banks distrust each other = bearish
        # Historical: <0.5% = normal, 0.5-1% = elevated, >1% = stress, 4.5% = 2008 crisis
        if value > 1.0:
            return "bearish"   # Bank funding stress = systemic risk elevated
        elif value < 0.3:
            return "bullish"   # Interbank market calm, banks lending freely
        return "neutral"

    elif series_id == "ICSA":
        # Initial Jobless Claims (seasonally adjusted, weekly, thousands of persons)
        # Low claims = tight labor market = bullish; rising claims = layoffs = bearish
        # Historical range: ~200K (tight labor) to 6M+ (COVID spike), ~300-400K = elevated
        if value < 220:
            return "bullish"   # Extremely tight labor market
        elif value > 350:
            return "bearish"   # Rising layoffs, labor market loosening
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
