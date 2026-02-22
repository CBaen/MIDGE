# Defense — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## ThreatDetector (threat_detector.py) — Multi-Strategy Defense

**Status:** BUILT. 431 lines. 4 biological defense strategies.

**Provides:**
- Porcupine: `register_quill()` + `scan_threats()` -> Proactive detection
- Turtle: `update_integrity()` -> Passive shell activation below threshold
- Lizard: `register_sacrificeable()` + `sacrifice()` -> Adaptive autotomy
- Kangaroo: `register_counter_action()` + `counterattack()` -> Active neutralization
- `respond_to_threat()` -> Automatic strategy selection based on threat level
- Energy system: defense costs energy, recharges over time

**EventBus channels (all WIRED):**
- `defense.threat_detected` / `defense.activated` / `defense.threat_neutralized`

**Cross-system connections:**

| System | Connection | Status |
|--------|-----------|--------|
| EventBus | Publishes threat/defense events | WIRED (via `Any` param) |
| HAVEN | `isolate_agent()` for kangaroo counterattack | BUILT (stub via `Any`) |

---

## InputValidator (input_validator.py) — Zero-Trust Boundary

**Status:** BUILT. 341 lines.

**Provides:**
- `validate(source, input_type, data)` -> Full validation pipeline
- Trust management: `get_trust()`, `update_trust()`, `is_trusted()`
- Anomaly detection: z-score based on input history
- Custom validators: `register_validator()`, `register_range_validator()`
- Convenience: `validate_policy_update()`, `validate_message()`, `validate_state_update()`

**EventBus channels (all WIRED):**
- `defense.validation_failed` / `defense.trust_updated`

---

## PearlDefense (pearl_defense.py) — Nacre-Inspired Encapsulation

**Status:** WIRED. Wraps InputValidator with alternating hard/soft validation layers.

**Provides:**
- `validate(source, input_type, data)` -> Multi-layer validation with encapsulation
- Threat encapsulation: suspicious inputs get wrapped in validation pearls
- Step hook for pearl lifecycle management (dissolution over time)

**EventBus channels (all WIRED):**
- `pearl.encapsulation_started` / `pearl.encapsulation_complete` / `pearl.dissolved`

---

## Related Modules

- [emergent/CONNECTIONS.md](../emergent/CONNECTIONS.md) — AutoHealer subscribes to defense events; SomaticMap updated on threat activation
- [learning/CONNECTIONS.md](../learning/CONNECTIONS.md) — InputValidator validates FRL policy updates
- [backbone/CONNECTIONS.md](../backbone/CONNECTIONS.md) — HAVEN risk alerts feed into ThreatDetector
