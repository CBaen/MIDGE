"""Curated causal chains for WorldModel — static seed data."""

from typing import List, Tuple


def _get_curated_chains() -> List[Tuple[str, str, dict]]:
    """Return curated causal relationships.

    Each tuple: (cause, effect, {strength, lag_days, direction, evidence})
    Direction refers to the effect on the downstream node:
    - "bearish" = cause event hurts the downstream node
    - "bullish" = cause event helps the downstream node
    """
    edges = []

    def _add(cause, effect, strength=0.7, lag=3.0, direction="bearish"):
        edges.append((cause, effect, {
            "strength": strength, "lag_days": lag,
            "direction": direction, "evidence": "curated",
            "hit_count": 0, "miss_count": 0,
        }))

    # === ENERGY CHAINS ===

    # Oil supply disruptions
    _add("hurricane_gulf", "oil_supply_disruption", 0.85, 1, "bearish")
    _add("opec_production_cut", "oil_supply_disruption", 0.90, 1, "bearish")
    _add("middle_east_conflict", "oil_supply_risk", 0.75, 1, "bearish")
    _add("oil_supply_risk", "crude_price_spike", 0.80, 2, "bullish")
    _add("oil_supply_disruption", "crude_price_spike", 0.85, 2, "bullish")

    # Crude price effects
    _add("crude_price_spike", "XLE", 0.85, 1, "bullish")
    _add("crude_price_spike", "XOP", 0.85, 1, "bullish")
    _add("crude_price_spike", "USO", 0.90, 0, "bullish")
    _add("crude_price_spike", "airline_fuel_costs", 0.90, 3, "bearish")
    _add("crude_price_spike", "chemical_feedstock_costs", 0.80, 5, "bearish")
    _add("crude_price_spike", "trucking_costs", 0.75, 7, "bearish")

    _add("airline_fuel_costs", "DAL", 0.80, 5, "bearish")
    _add("airline_fuel_costs", "UAL", 0.80, 5, "bearish")
    _add("airline_fuel_costs", "AAL", 0.80, 5, "bearish")
    _add("airline_fuel_costs", "LUV", 0.75, 5, "bearish")

    _add("chemical_feedstock_costs", "DOW", 0.70, 10, "bearish")
    _add("chemical_feedstock_costs", "LYB", 0.70, 10, "bearish")

    # Natural gas chains
    _add("nat_gas_spike", "fertilizer_costs", 0.75, 7, "bearish")
    _add("nat_gas_spike", "UNG", 0.90, 0, "bullish")
    _add("fertilizer_costs", "agricultural_costs", 0.70, 14, "bearish")
    _add("agricultural_costs", "ADM", 0.60, 21, "bearish")
    _add("agricultural_costs", "DE", 0.55, 21, "bullish")  # farm equipment demand

    # EIA inventory signals
    _add("eia_crude_build", "crude_price_pressure_down", 0.70, 1, "bearish")
    _add("crude_price_pressure_down", "XLE", 0.65, 1, "bearish")
    _add("eia_crude_draw", "crude_price_pressure_up", 0.70, 1, "bullish")
    _add("crude_price_pressure_up", "XLE", 0.65, 1, "bullish")

    # === MONETARY / MACRO CHAINS ===

    # Fed rate decisions
    _add("fed_rate_hike", "borrowing_costs_up", 0.95, 1, "bearish")
    _add("fed_rate_hike", "dollar_strengthens", 0.80, 1, "bullish")
    _add("fed_rate_hike", "bank_margins_up", 0.80, 5, "bullish")

    _add("borrowing_costs_up", "housing_demand_down", 0.80, 14, "bearish")
    _add("borrowing_costs_up", "growth_stocks_pressure", 0.75, 3, "bearish")
    _add("housing_demand_down", "XHB", 0.80, 7, "bearish")
    _add("housing_demand_down", "ITB", 0.80, 7, "bearish")
    _add("growth_stocks_pressure", "QQQ", 0.70, 3, "bearish")
    _add("growth_stocks_pressure", "ARKK", 0.80, 3, "bearish")

    _add("bank_margins_up", "XLF", 0.75, 5, "bullish")
    _add("bank_margins_up", "KRE", 0.80, 5, "bullish")

    _add("dollar_strengthens", "export_competitiveness_down", 0.70, 14, "bearish")
    _add("export_competitiveness_down", "multinational_earnings_pressure", 0.65, 30, "bearish")

    # Inflation chains
    _add("cpi_hot", "rate_hike_expectation", 0.85, 1, "bearish")
    _add("rate_hike_expectation", "growth_stocks_pressure", 0.75, 1, "bearish")
    _add("rate_hike_expectation", "bond_prices_down", 0.85, 0, "bearish")
    _add("bond_prices_down", "TLT", 0.90, 0, "bearish")

    # Labor market
    _add("unemployment_spike", "consumer_spending_down", 0.80, 14, "bearish")
    _add("consumer_spending_down", "XRT", 0.75, 7, "bearish")
    _add("consumer_spending_down", "XLY", 0.75, 7, "bearish")

    _add("nfp_strong", "consumer_confidence_up", 0.70, 3, "bullish")
    _add("consumer_confidence_up", "XLY", 0.65, 5, "bullish")

    # === TECHNOLOGY / SEMICONDUCTOR CHAINS ===

    _add("china_taiwan_tension", "chip_supply_risk", 0.80, 1, "bearish")
    _add("chip_supply_risk", "semiconductor_prices_up", 0.75, 7, "bullish")
    _add("semiconductor_prices_up", "TSM", 0.80, 3, "bullish")
    _add("semiconductor_prices_up", "auto_production_disruption", 0.70, 14, "bearish")

    _add("auto_production_disruption", "F", 0.70, 14, "bearish")
    _add("auto_production_disruption", "GM", 0.70, 14, "bearish")
    _add("auto_production_disruption", "TM", 0.65, 14, "bearish")

    _add("ai_capex_surge", "datacenter_demand", 0.85, 7, "bullish")
    _add("datacenter_demand", "NVDA", 0.90, 5, "bullish")
    _add("datacenter_demand", "AMD", 0.80, 5, "bullish")
    _add("datacenter_demand", "electricity_demand_up", 0.75, 30, "bullish")
    _add("electricity_demand_up", "utility_revenue_up", 0.70, 30, "bullish")
    _add("utility_revenue_up", "XLU", 0.65, 14, "bullish")

    # === DEFENSE / GOVERNMENT ===

    _add("defense_spending_increase", "defense_contract_awards", 0.85, 30, "bullish")
    _add("defense_contract_awards", "LMT", 0.80, 14, "bullish")
    _add("defense_contract_awards", "RTX", 0.80, 14, "bullish")
    _add("defense_contract_awards", "NOC", 0.80, 14, "bullish")
    _add("defense_contract_awards", "GD", 0.75, 14, "bullish")

    _add("geopolitical_tension_escalation", "defense_spending_increase", 0.75, 30, "bullish")
    _add("geopolitical_tension_escalation", "gold_demand_up", 0.80, 3, "bullish")
    _add("gold_demand_up", "GLD", 0.85, 1, "bullish")
    _add("gold_demand_up", "GDX", 0.80, 1, "bullish")

    # === HEALTHCARE / PHARMA ===

    _add("fda_approval", "drug_revenue_growth", 0.85, 1, "bullish")
    _add("fda_rejection", "drug_pipeline_setback", 0.90, 0, "bearish")

    _add("healthcare_reform_legislation", "pharma_pricing_pressure", 0.70, 30, "bearish")
    _add("pharma_pricing_pressure", "XLV", 0.65, 14, "bearish")

    # === SUPPLY CHAIN ===

    _add("port_strike", "shipping_disruption", 0.90, 1, "bearish")
    _add("shipping_disruption", "inventory_shortage", 0.80, 14, "bearish")
    _add("inventory_shortage", "retail_margin_pressure", 0.70, 21, "bearish")
    _add("retail_margin_pressure", "WMT", 0.60, 14, "bearish")
    _add("retail_margin_pressure", "TGT", 0.65, 14, "bearish")

    _add("drought_major", "crop_yield_down", 0.80, 30, "bearish")
    _add("crop_yield_down", "food_prices_up", 0.75, 14, "bullish")
    _add("food_prices_up", "DBA", 0.80, 3, "bullish")

    _add("steel_tariffs", "construction_costs_up", 0.75, 14, "bearish")
    _add("construction_costs_up", "housing_starts_down", 0.70, 21, "bearish")
    _add("housing_starts_down", "XHB", 0.70, 14, "bearish")

    # === REGULATORY / ANTITRUST ===

    _add("antitrust_action_tech", "market_power_reduced", 0.70, 30, "bearish")
    _add("market_power_reduced", "mega_tech_pressure", 0.65, 14, "bearish")

    _add("crypto_regulation_positive", "crypto_adoption_up", 0.75, 7, "bullish")
    _add("crypto_regulation_negative", "crypto_selloff", 0.80, 1, "bearish")

    # === GEOPOLITICAL ===

    _add("russia_sanctions", "energy_supply_disruption_eu", 0.80, 3, "bearish")
    _add("energy_supply_disruption_eu", "nat_gas_spike", 0.85, 1, "bullish")

    _add("china_tariffs", "import_costs_up", 0.80, 14, "bearish")
    _add("import_costs_up", "consumer_prices_up", 0.70, 30, "bearish")
    _add("consumer_prices_up", "cpi_hot", 0.75, 30, "bearish")

    # === CROSS-DOMAIN AMPLIFIERS ===
    # These connect domains that can create feedback loops

    _add("vix_spike", "risk_off_rotation", 0.80, 0, "bearish")
    _add("risk_off_rotation", "growth_stocks_pressure", 0.75, 1, "bearish")
    _add("risk_off_rotation", "gold_demand_up", 0.70, 1, "bullish")
    _add("risk_off_rotation", "bond_prices_up", 0.75, 0, "bullish")
    _add("bond_prices_up", "TLT", 0.85, 0, "bullish")

    _add("credit_spread_widening", "recession_fear", 0.75, 7, "bearish")
    _add("recession_fear", "consumer_spending_down", 0.70, 14, "bearish")
    _add("recession_fear", "XLF", 0.70, 3, "bearish")

    # === CRYPTO CHAINS ===
    # BTC correlates with Nasdaq/risk assets — regime dependent

    # Oil → macro → crypto: oil spikes = inflation fear = Fed hawkish = crypto down
    _add("crude_price_spike", "inflation_expectation_up", 0.75, 3, "bearish")
    _add("inflation_expectation_up", "fed_hawkish_signal", 0.70, 7, "bearish")
    _add("fed_hawkish_signal", "BTC/USD", 0.70, 2, "bearish")
    _add("fed_hawkish_signal", "ETH/USD", 0.65, 2, "bearish")

    # Risk-off → crypto down (BTC moves with Nasdaq in correlated regime)
    _add("risk_off_rotation", "BTC/USD", 0.70, 1, "bearish")
    _add("risk_off_rotation", "ETH/USD", 0.65, 1, "bearish")

    # VIX spike → crypto selloff (fear propagation)
    _add("vix_spike", "BTC/USD", 0.65, 0, "bearish")

    # Asian markets → crypto (Hong Kong BTC ETF linkage)
    _add("asia_market_selloff", "BTC/USD", 0.60, 0.17, "bearish")  # 0.17 days = ~4 hours

    # Crypto regulation events
    _add("crypto_regulation_positive", "BTC/USD", 0.75, 3, "bullish")
    _add("crypto_regulation_negative", "BTC/USD", 0.80, 1, "bearish")
    _add("crypto_adoption_up", "BTC/USD", 0.70, 7, "bullish")
    _add("crypto_selloff", "BTC/USD", 0.85, 0, "bearish")
    _add("crypto_selloff", "ETH/USD", 0.85, 0, "bearish")

    # Stablecoin flows → crypto direction
    _add("stablecoin_minting", "BTC/USD", 0.65, 2, "bullish")
    _add("stablecoin_burning", "BTC/USD", 0.65, 2, "bearish")

    # DeFi capital flight → crypto bearish
    _add("defi_tvl_drop", "ETH/USD", 0.70, 1, "bearish")
    _add("defi_tvl_drop", "BTC/USD", 0.55, 2, "bearish")

    # War/geopolitical → defense stocks + energy + crypto safe-haven narrative
    _add("middle_east_conflict", "defense_spending_up", 0.80, 7, "bullish")
    _add("defense_spending_up", "LMT", 0.80, 3, "bullish")
    _add("defense_spending_up", "RTX", 0.80, 3, "bullish")
    _add("defense_spending_up", "NOC", 0.75, 3, "bullish")
    _add("middle_east_conflict", "BTC/USD", 0.45, 7, "bullish")  # weak safe-haven, regime dependent

    return edges
