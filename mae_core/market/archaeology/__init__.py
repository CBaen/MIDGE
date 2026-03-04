"""Pattern Archaeology — reverse-engineering historical market moves.

MIDGE's pattern discovery system. She doesn't wait. She doesn't stop.
She continuously digs through history, discovers patterns, cross-validates
them across the entire market, and watches for those patterns recurring.

Three levels of abstraction:
  - MoveFingerprint: what preceded ONE specific price move on ONE symbol
  - PatternTemplate: the symbol-AGNOSTIC pattern seen across MANY symbols
  - PatternStack: multiple independent templates converging on one ticker NOW

Components:
  - fingerprint: MoveFingerprint + PrecursorSignal + PatternTemplate dataclasses
  - dig_sites: Automatic dig site generation from price history
  - historical_fetcher: Active historical data retrieval (TA, SEC, FRED, COT, Congressional)
  - excavator: Reverse-engineer historical moves using ALL data sources
  - pattern_library: Store/query fingerprints AND symbol-agnostic templates
  - pattern_watcher: Live template-based stacking detection engine
  - excavation_daemon: Continuous background pattern discovery (replaces CLI)
"""
