"""Pattern Archaeology — reverse-engineering historical market moves.

MIDGE's pattern discovery system. Takes known price moves that already
happened, works backward through the signal archive to find what preceded
them, builds a library of "fingerprints," and watches the live market
for those fingerprints recurring and stacking.

Components:
  - fingerprint: MoveFingerprint + PrecursorSignal dataclasses
  - dig_sites: Automatic dig site generation from price history
  - excavator: Reverse-engineer historical moves into fingerprints
  - pattern_library: Store/query/manage discovered fingerprints
  - pattern_watcher: Live stacking detection engine
"""
