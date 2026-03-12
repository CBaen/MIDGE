"""fred_models.py - FRED data models and direction classification logic.

Extracted from fred_client.py. Contains:
  - FRED_SERIES: series ID -> (name, signal_type) mapping
  - DELTA_SERIES: series IDs that require two observations for directional signal
  - MacroIndicator: dataclass for a single FRED observation
  - _determine_direction: classify a macro value as bullish/bearish/neutral
  - _determine_direction_from_delta: classify based on month-over-month change
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Series that report raw index LEVELS — direction is only meaningful as
# month-over-month or year-over-year CHANGE.  get_series() fetches limit=2
# for these so prior_value is always available for delta computation.
#
# Semantics per series:
#   PCEPI / PCEPILFE / CPIAUCSL — inflation indices: rising = bearish (price pressure)
#   RSXFS  — retail sales: rising = bullish (consumer spending growing)
#   M2SL   — money supply: rising = bullish (liquidity expanding)
#   DFF    — fed funds rate: rising = bearish (tighter conditions)
#   DGS2 / DGS10 — treasury yields: rising = bearish for equities (tighter)
DELTA_SERIES: set[str] = {
    "PCEPI", "PCEPILFE", "CPIAUCSL",  # inflation price indices
    "RSXFS",                            # retail sales level
    "M2SL",                             # money supply level
    "DFF",                              # federal funds rate level
    "DGS2", "DGS10",                    # treasury yield levels
}

# For each delta series: True = rising is bullish, False = rising is bearish.
# Thresholds (in %) below which the change is too small to signal.
_DELTA_BULLISH_IF_RISING: dict[str, bool] = {
    "PCEPI":    False,  # higher inflation = bearish
    "PCEPILFE": False,  # higher core inflation = bearish
    "CPIAUCSL": False,  # higher CPI = bearish
    "RSXFS":    True,   # more retail spending = bullish
    "M2SL":     True,   # more money supply = bullish liquidity
    "DFF":      False,  # higher rate = tighter = bearish for risk
    "DGS2":     False,  # higher 2Y yield = tighter = bearish for equities
    "DGS10":    False,  # higher 10Y yield = tighter = bearish for equities
}

# Minimum absolute percent change (|change_pct|) to break out of neutral.
# Avoids noise on near-flat months.
_DELTA_NEUTRAL_BAND: dict[str, float] = {
    "PCEPI":    0.05,   # 0.05% mom = meaningful price move
    "PCEPILFE": 0.05,
    "CPIAUCSL": 0.05,
    "RSXFS":    0.20,   # retail sales need 0.2% mom to signal (volatile series)
    "M2SL":     0.10,   # 0.1% mom M2 change = meaningful
    "DFF":      0.10,   # 10 bps fed funds move = meaningful
    "DGS2":     0.05,   # 5 bps 2Y yield move = meaningful
    "DGS10":    0.05,   # 5 bps 10Y yield move = meaningful
}


def _determine_direction_from_delta(series_id: str, change_pct: float) -> str:
    """Classify direction for level-based series using month-over-month change.

    Args:
        series_id: FRED series identifier (must be in DELTA_SERIES)
        change_pct: Percent change from prior observation to current
                    ((current - prior) / prior * 100)

    Returns:
        "bullish", "bearish", or "neutral"
    """
    neutral_band = _DELTA_NEUTRAL_BAND.get(series_id, 0.05)
    if abs(change_pct) < neutral_band:
        return "neutral"

    rising_is_bullish = _DELTA_BULLISH_IF_RISING.get(series_id, True)
    if change_pct > 0:
        return "bullish" if rising_is_bullish else "bearish"
    else:
        return "bearish" if rising_is_bullish else "bullish"


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


def _determine_direction(
    series_id: str,
    value: float,
    change_pct: Optional[float] = None,
) -> str:
    """Classify a macro value as bullish, bearish, or neutral.

    For level-based series (those in DELTA_SERIES), direction is computed from
    the month-over-month percent change rather than the raw level.  When
    change_pct is None (only one observation available), these series fall back
    to "neutral" rather than guessing from the level.

    Rules are based on historically significant thresholds — not precise
    predictions, but reliable regime signals.

    Args:
        series_id: FRED series identifier
        value: Numeric value of the observation (used for threshold-based series)
        change_pct: Percent change from prior observation, if available.
                    Required for DELTA_SERIES to produce non-neutral direction.

    Returns:
        "bullish", "bearish", or "neutral"
    """
    # --- Delta-based series: direction from change, not level ---
    if series_id in DELTA_SERIES:
        if change_pct is None:
            return "neutral"
        return _determine_direction_from_delta(series_id, change_pct)
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

    elif series_id == "UMCSENT":
        # University of Michigan Consumer Sentiment (index, 1966=100 baseline)
        # Historical range: ~55 (troughs: 2008, 2022) to ~100+ (expansions)
        if value >= 85:
            return "bullish"   # Strong sentiment = consumer spending confidence
        elif value <= 65:
            return "bearish"   # Weak sentiment = consumers pulling back
        return "neutral"

    elif series_id == "T5YIE":
        # 5-Year Breakeven Inflation Rate (market-implied, %)
        # At or above 3% = inflation expectations dangerously unanchored = hawkish risk = bearish
        # Below 2% = inflation anchored, Fed not forced to hike = bullish for risk assets
        if value >= 3.0:
            return "bearish"   # Inflation expectations at or above danger threshold
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

    For level-based series (those in DELTA_SERIES), prior_value and
    change_pct are populated from the second observation fetched by
    get_series().  direction is derived from change_pct rather than the
    raw level in those cases.
    """

    series_id: str          # e.g. "T10Y2Y"
    series_name: str        # e.g. "10Y-2Y Treasury Spread (Yield Curve)"
    value: float            # Numeric observation value (current)
    date: str               # YYYY-MM-DD of the observation
    signal_type: str        # "yield_curve", "credit_spread", "volatility", "rates", "employment", "inflation"
    direction: str          # "bullish", "bearish", or "neutral"
    signal_source: str = "fred_macro"
    decay_rate: float = 0.01    # Slow decay — macro regimes persist ~70 days
    confidence: float = 0.70    # Government data is reliable but lags reality
    # Delta fields — populated for DELTA_SERIES only
    prior_value: Optional[float] = None    # Previous observation value
    change_pct: Optional[float] = None     # ((value - prior_value) / prior_value) * 100
