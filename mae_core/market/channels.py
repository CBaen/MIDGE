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
CH_OUTCOME_OBSERVED = "market.sensing.outcome_observed"
CH_PREDICTION_RESULT = "market.sensing.prediction_result"
