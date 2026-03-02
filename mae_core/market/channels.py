"""EventBus channel constants for MIDGE market intelligence.

All market-related EventBus channels defined here. No magic strings elsewhere.
"""

# Market Edge channels (published by edge detectors)
CH_CLUSTER_DETECTED = "market.edge.cluster_detected"
CH_POLITICIAN_TRADE = "market.edge.politician_trade"
CH_FILING_ANOMALY = "market.edge.filing_anomaly"
CH_CONTRACT_PREDICTED = "market.edge.contract_predicted"
CH_SESSION_SWEEP = "market.edge.session_sweep"

# Market Intel channels (published by intelligence layer)
CH_VELOCITY_ANOMALY = "market.intel.velocity_anomaly"
CH_CONVERGENCE = "market.intel.convergence"
CH_ACTIONABLE = "market.intel.actionable"
CH_THOMPSON_STATS = "market.intel.thompson_stats"
CH_LAG_FINDING = "market.intel.lag_finding"
CH_KELLY_SIZING = "market.intel.kelly_sizing"

# Market Sensing channels (ingest and feedback)
CH_SIGNAL_RECEIVED = "market.sensing.signal_received"
CH_SIGNAL_INGESTED = "market.sensing.signal_ingested"
CH_OUTCOME_OBSERVED = "market.sensing.outcome_observed"
CH_PREDICTION_RESULT = "market.sensing.prediction_result"

# Hypothesis lifecycle channels (RSI Layer 2)
CH_HYPOTHESIS_DISCOVERED = "market.hypothesis.discovered"
CH_HYPOTHESIS_PROMOTED = "market.hypothesis.promoted"
CH_HYPOTHESIS_RETIRED = "market.hypothesis.retired"
CH_HYPOTHESIS_FIRED = "market.hypothesis.fired"

# Backtest bridge channels
CH_BACKTEST_REFRESHED = "market.intel.backtest_refreshed"

# Meta-learning channels (RSI Layer 3 — Bridges 4 & 5)
CH_GATE_ADJUSTED = "market.meta.gate_adjusted"
CH_META_ADJUSTED = "market.meta.meta_adjusted"

# Coherence channels (narrative conflict detection — Capability 3)
CH_CONTRADICTION_DETECTED = "market.intel.contradiction_detected"

# Absence detection channels (the dog that didn't bark)
CH_ABSENCE_DETECTED = "market.intel.absence_detected"
