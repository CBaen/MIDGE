"""KnowledgeGraph — persistent Neo4j store for all MIDGE learned knowledge.

Phase 3 of Three Conditions for Inevitability. Every learned fact
(Thompson updates, Granger causality, convergence alerts, outcomes,
cascade confirmations) is mirrored to Neo4j alongside the flat files.

Dual-write pattern:
  - Flat files remain the authoritative fallback — if Neo4j is down, MIDGE
    keeps running without interruption.
  - Neo4j is the queryable truth — graph traversal for causal chains,
    ticker histories, and learning trajectories that flat files cannot answer.
  - If flat files are corrupted, Neo4j holds the reconstruction source.

Schema (created on first connect, idempotent):
  Nodes:
    (:Domain {name})            — signal domain (e.g. "insider", "macro")
    (:Ticker {symbol})          — tradeable instrument
    (:Source {key})             — Thompson signal source key

  Relationships:
    (:Domain)-[:CAUSES {lag, f_stat, p_value, discovered_at}]->(:Domain)
    (:Source)-[:THOMPSON_UPDATE {alpha, beta, won, regime, timestamp}]->(:Source)
    (:Ticker)-[:CONVERGENCE_ALERT {alert_id, direction, confidence, domains, timestamp}]->(:Ticker)
    (:Ticker)-[:OUTCOME {prediction_id, won, return_pct, timestamp}]->(:Ticker)
    (:Ticker)-[:CASCADE_TO {lag, confirmed, energy_ratio, cascade_id, timestamp}]->(:Ticker)

Note: THOMPSON_UPDATE is a self-loop (source → same source). Neo4j supports
this natively and it allows efficient range queries on the update history.

NEVER call this from hot paths. All write methods are best-effort: they wrap
in try/except and return silently on failure.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Environment variable names — matches the Docker compose setup.
_ENV_URI  = "MIDGE_NEO4J_URI"
_ENV_USER = "MIDGE_NEO4J_USER"
_ENV_PASS = "MIDGE_NEO4J_PASS"

_DEFAULT_URI  = "bolt://localhost:7687"
_DEFAULT_USER = "neo4j"
_DEFAULT_PASS = "midgepassword"

# Cypher: idempotent schema initialisation (constraints + indexes).
_SCHEMA_CYPHER = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Ticker) REQUIRE t.symbol IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.key IS UNIQUE",
    # Index relationship timestamps for range queries
    "CREATE INDEX IF NOT EXISTS FOR ()-[r:CAUSES]-() ON (r.discovered_at)",
    "CREATE INDEX IF NOT EXISTS FOR ()-[r:CONVERGENCE_ALERT]-() ON (r.timestamp)",
    "CREATE INDEX IF NOT EXISTS FOR ()-[r:OUTCOME]-() ON (r.timestamp)",
    "CREATE INDEX IF NOT EXISTS FOR ()-[r:THOMPSON_UPDATE]-() ON (r.timestamp)",
]


class KnowledgeGraph:
    """Persistent Neo4j knowledge store for MIDGE.

    All public methods are safe to call when Neo4j is unavailable —
    they check ``is_connected()`` and return ``None`` / empty lists
    silently rather than raising.

    Thread safety: the neo4j driver maintains its own connection pool;
    each session is acquired/released per call so concurrent writers
    are safe.
    """

    def __init__(
        self,
        uri: str = _DEFAULT_URI,
        user: str = _DEFAULT_USER,
        password: str = _DEFAULT_PASS,
    ) -> None:
        # Prefer environment variables so the daemon can be reconfigured
        # without touching source code.
        self._uri  = os.environ.get(_ENV_URI,  uri)
        self._user = os.environ.get(_ENV_USER, user)
        self._pass = os.environ.get(_ENV_PASS, password)
        self._driver = None
        self._connected = False
        self._connect()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Attempt to connect to Neo4j and initialise schema."""
        try:
            from neo4j import GraphDatabase  # type: ignore
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._pass),
            )
            # Verify connectivity with a lightweight ping
            self._driver.verify_connectivity()
            self._connected = True
            logger.info("KnowledgeGraph: connected to Neo4j at %s", self._uri)
            self._init_schema()
        except Exception:
            logger.warning(
                "KnowledgeGraph: Neo4j unavailable at %s — running on flat-file fallback.",
                self._uri,
                exc_info=False,
            )
            self._driver = None
            self._connected = False

    def is_connected(self) -> bool:
        """Return True if Neo4j is available and the session is live."""
        return self._connected and self._driver is not None

    def close(self) -> None:
        """Close the driver connection pool (call on daemon shutdown)."""
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
            self._connected = False

    def _init_schema(self) -> None:
        """Create constraints and indexes (idempotent — safe to re-run)."""
        if not self.is_connected():
            return
        try:
            with self._driver.session() as session:
                for stmt in _SCHEMA_CYPHER:
                    try:
                        session.run(stmt)
                    except Exception:
                        logger.debug("KnowledgeGraph schema stmt failed: %s", stmt, exc_info=True)
            logger.debug("KnowledgeGraph: schema initialised")
        except Exception:
            logger.debug("KnowledgeGraph: schema init failed", exc_info=True)

    def _run(self, cypher: str, params: dict) -> Optional[List[Dict[str, Any]]]:
        """Execute a Cypher statement and return rows, or None on failure."""
        if not self.is_connected():
            return None
        try:
            with self._driver.session() as session:
                result = session.run(cypher, **params)
                return [dict(r) for r in result]
        except Exception:
            logger.debug("KnowledgeGraph Cypher failed: %.120s", cypher, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Write methods — dual-write helpers
    # ------------------------------------------------------------------

    def store_thompson_update(
        self,
        source_key: str,
        regime: str,
        alpha: float,
        beta: float,
        won: bool,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Persist a Thompson distribution update as a self-loop on the Source node.

        Args:
            source_key: Signal source identifier (e.g. "insider_form4").
            regime: Market regime at update time (e.g. "bull", "default").
            alpha: Beta distribution alpha after update.
            beta:  Beta distribution beta after update.
            won:   Whether the associated prediction was correct.
            timestamp: Defaults to now.
        """
        if not self.is_connected():
            return
        ts = (timestamp or datetime.now()).isoformat()
        cypher = """
        MERGE (s:Source {key: $key})
        CREATE (s)-[:THOMPSON_UPDATE {
            alpha:     $alpha,
            beta:      $beta,
            won:       $won,
            regime:    $regime,
            timestamp: $ts
        }]->(s)
        """
        self._run(cypher, {
            "key": source_key,
            "alpha": alpha,
            "beta": beta,
            "won": won,
            "regime": regime,
            "ts": ts,
        })

    def store_granger_finding(
        self,
        cause_domain: str,
        effect_domain: str,
        lag: int,
        f_stat: float,
        p_value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Persist a Granger causal relationship between two Domain nodes.

        If the CAUSES edge already exists, it is updated in-place so the
        graph always holds the most recent finding rather than accumulating
        stale duplicates.

        Args:
            cause_domain:  Domain that Granger-causes the effect.
            effect_domain: Domain that is Granger-caused.
            lag:           Best lag (days) with strongest evidence.
            f_stat:        F-test statistic at best lag.
            p_value:       p-value at best lag.
            timestamp:     Defaults to now.
        """
        if not self.is_connected():
            return
        ts = (timestamp or datetime.now()).isoformat()
        cypher = """
        MERGE (c:Domain {name: $cause})
        MERGE (e:Domain {name: $effect})
        MERGE (c)-[r:CAUSES]->(e)
        SET r.lag          = $lag,
            r.f_stat       = $f_stat,
            r.p_value      = $p_value,
            r.discovered_at = $ts
        """
        self._run(cypher, {
            "cause": cause_domain,
            "effect": effect_domain,
            "lag": lag,
            "f_stat": f_stat,
            "p_value": p_value,
            "ts": ts,
        })

    def store_convergence_alert(
        self,
        alert_id: str,
        ticker: str,
        direction: str,
        confidence: float,
        domains: List[str],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Persist a convergence alert as a self-loop on the Ticker node.

        Alerts are self-loops (ticker → same ticker) so all alerts for a
        symbol are queryable via ``MATCH (t:Ticker {symbol:$sym})-[:CONVERGENCE_ALERT]->(t)``.

        Args:
            alert_id:   Unique identifier for deduplication.
            ticker:     Instrument symbol (e.g. "AAPL").
            direction:  "bullish" or "bearish".
            confidence: Thompson-weighted confidence score (0-1).
            domains:    List of converging domain names.
            timestamp:  Defaults to now.
        """
        if not self.is_connected():
            return
        ts = (timestamp or datetime.now()).isoformat()
        cypher = """
        MERGE (t:Ticker {symbol: $symbol})
        CREATE (t)-[:CONVERGENCE_ALERT {
            alert_id:   $alert_id,
            direction:  $direction,
            confidence: $confidence,
            domains:    $domains,
            timestamp:  $ts
        }]->(t)
        """
        self._run(cypher, {
            "symbol": ticker,
            "alert_id": alert_id,
            "direction": direction,
            "confidence": confidence,
            "domains": domains,
            "ts": ts,
        })

    def store_outcome(
        self,
        prediction_id: str,
        ticker: str,
        won: bool,
        return_pct: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Persist an outcome grade linked to a prediction self-loop on Ticker.

        Args:
            prediction_id: ID matching the original convergence alert or signal.
            ticker:        Instrument symbol.
            won:           Whether the prediction was correct.
            return_pct:    Actual percentage return (positive = price went up).
            timestamp:     Defaults to now.
        """
        if not self.is_connected():
            return
        ts = (timestamp or datetime.now()).isoformat()
        cypher = """
        MERGE (t:Ticker {symbol: $symbol})
        CREATE (t)-[:OUTCOME {
            prediction_id: $prediction_id,
            won:           $won,
            return_pct:    $return_pct,
            timestamp:     $ts
        }]->(t)
        """
        self._run(cypher, {
            "symbol": ticker,
            "prediction_id": prediction_id,
            "won": won,
            "return_pct": return_pct,
            "ts": ts,
        })

    def store_cascade_confirmation(
        self,
        cascade_id: str,
        from_ticker: str,
        to_ticker: str,
        lag: float,
        confirmed: bool,
        energy_ratio: float = 1.0,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Persist a cascade domino link as a directed edge between Tickers.

        Args:
            cascade_id:    Source convergence alert ID.
            from_ticker:   The trigger ticker that fired first.
            to_ticker:     The predicted downstream ticker.
            lag:           Expected lag in days between the two.
            confirmed:     True = domino fell; False = expired unconfirmed.
            energy_ratio:  Actual lag / predicted lag (>1 = faster; <1 = slower).
            timestamp:     Defaults to now.
        """
        if not self.is_connected():
            return
        ts = (timestamp or datetime.now()).isoformat()
        cypher = """
        MERGE (a:Ticker {symbol: $from_sym})
        MERGE (b:Ticker {symbol: $to_sym})
        CREATE (a)-[:CASCADE_TO {
            cascade_id:   $cascade_id,
            lag:          $lag,
            confirmed:    $confirmed,
            energy_ratio: $energy_ratio,
            timestamp:    $ts
        }]->(b)
        """
        self._run(cypher, {
            "from_sym": from_ticker,
            "to_sym": to_ticker,
            "cascade_id": cascade_id,
            "lag": lag,
            "confirmed": confirmed,
            "energy_ratio": energy_ratio,
            "ts": ts,
        })

    # ------------------------------------------------------------------
    # Read methods — graph traversal queries
    # ------------------------------------------------------------------

    def get_ticker_history(
        self,
        ticker: str,
        limit: int = 50,
    ) -> Dict[str, List[dict]]:
        """Return all alerts, outcomes, and cascade relationships for a ticker.

        Useful for generating a narrative about an instrument: what patterns
        MIDGE has seen, what outcomes were graded, and what cascade chains
        it has been part of.

        Args:
            ticker: Instrument symbol.
            limit:  Maximum rows per result type.

        Returns:
            {
              "alerts":   [{alert_id, direction, confidence, domains, timestamp}],
              "outcomes": [{prediction_id, won, return_pct, timestamp}],
              "cascades": [{cascade_id, to_sym, lag, confirmed, energy_ratio, timestamp}],
            }
        """
        if not self.is_connected():
            return {"alerts": [], "outcomes": [], "cascades": []}

        alerts_cypher = """
        MATCH (t:Ticker {symbol: $symbol})-[r:CONVERGENCE_ALERT]->(t)
        RETURN r.alert_id AS alert_id, r.direction AS direction,
               r.confidence AS confidence, r.domains AS domains,
               r.timestamp AS timestamp
        ORDER BY r.timestamp DESC LIMIT $limit
        """
        outcomes_cypher = """
        MATCH (t:Ticker {symbol: $symbol})-[r:OUTCOME]->(t)
        RETURN r.prediction_id AS prediction_id, r.won AS won,
               r.return_pct AS return_pct, r.timestamp AS timestamp
        ORDER BY r.timestamp DESC LIMIT $limit
        """
        cascades_cypher = """
        MATCH (t:Ticker {symbol: $symbol})-[r:CASCADE_TO]->(b:Ticker)
        RETURN r.cascade_id AS cascade_id, b.symbol AS to_sym,
               r.lag AS lag, r.confirmed AS confirmed,
               r.energy_ratio AS energy_ratio, r.timestamp AS timestamp
        ORDER BY r.timestamp DESC LIMIT $limit
        """
        params = {"symbol": ticker, "limit": limit}
        return {
            "alerts":   self._run(alerts_cypher,   params) or [],
            "outcomes": self._run(outcomes_cypher,  params) or [],
            "cascades": self._run(cascades_cypher,  params) or [],
        }

    def get_causal_chain(
        self,
        start_domain: str,
        max_hops: int = 3,
    ) -> List[dict]:
        """Traverse the Granger causal graph from a Domain node.

        Returns domain-to-domain paths (as lists of {cause, effect, lag,
        f_stat, p_value}) ordered by cumulative f_stat descending so the
        strongest causal chains surface first.

        Args:
            start_domain: Starting domain name (e.g. "institutional").
            max_hops:     Maximum causal hops to traverse.

        Returns:
            List of path dicts with keys: path_nodes, lags, f_stats, p_values.
        """
        if not self.is_connected():
            return []
        cypher = """
        MATCH p = (start:Domain {name: $domain})-[:CAUSES*1..$hops]->(end:Domain)
        UNWIND relationships(p) AS r
        WITH p,
             collect(startNode(r).name) + [endNode(last(relationships(p))).name] AS nodes,
             collect(r.lag) AS lags,
             collect(r.f_stat) AS f_stats,
             collect(r.p_value) AS p_values,
             reduce(s = 0.0, x IN collect(r.f_stat) | s + x) AS cum_f
        RETURN nodes AS path_nodes, lags, f_stats, p_values, cum_f
        ORDER BY cum_f DESC
        LIMIT 20
        """
        rows = self._run(cypher, {"domain": start_domain, "hops": max_hops})
        return rows or []

    def get_learning_trajectory(
        self,
        source_key: str,
        limit: int = 100,
    ) -> List[dict]:
        """Return Thompson update history for a source — the learning log.

        Ordered oldest-first so callers can plot the belief evolution over
        time: how alpha/beta grew, regime at each update, and win/loss run.

        Args:
            source_key: Signal source identifier.
            limit:      Maximum updates to return.

        Returns:
            List of {alpha, beta, won, regime, timestamp} dicts.
        """
        if not self.is_connected():
            return []
        cypher = """
        MATCH (s:Source {key: $key})-[r:THOMPSON_UPDATE]->(s)
        RETURN r.alpha AS alpha, r.beta AS beta,
               r.won AS won, r.regime AS regime,
               r.timestamp AS timestamp
        ORDER BY r.timestamp ASC LIMIT $limit
        """
        rows = self._run(cypher, {"key": source_key, "limit": limit})
        return rows or []

    # ------------------------------------------------------------------
    # Seed from existing flat files (one-time migration)
    # ------------------------------------------------------------------

    def seed_from_granger(self, granger_path: Optional[Path] = None) -> int:
        """Migrate existing Granger findings from flat file to Neo4j.

        Safe to re-run — MERGE semantics prevent duplicates. Reads from
        ``data/market/granger_continuous.json`` by default.

        Returns:
            Number of findings successfully written.
        """
        if not self.is_connected():
            return 0
        if granger_path is None:
            granger_path = (
                Path(__file__).resolve().parents[3]
                / "data" / "market" / "granger_continuous.json"
            )
        if not granger_path.exists():
            logger.debug("KnowledgeGraph: Granger seed file not found at %s", granger_path)
            return 0
        try:
            data = json.loads(granger_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("KnowledgeGraph: could not read Granger file", exc_info=True)
            return 0

        count = 0
        for finding in data.get("findings", []):
            try:
                self.store_granger_finding(
                    cause_domain=finding["cause_domain"],
                    effect_domain=finding["effect_domain"],
                    lag=finding.get("best_lag", 0),
                    f_stat=finding.get("f_statistic", 0.0),
                    p_value=finding.get("p_value", 1.0),
                    timestamp=datetime.fromisoformat(
                        finding.get("last_updated", datetime.now().isoformat())
                    ),
                )
                count += 1
            except Exception:
                logger.debug("KnowledgeGraph: Granger finding seed failed", exc_info=True)

        logger.info("KnowledgeGraph: seeded %d Granger findings", count)
        return count

    def seed_from_thompson(self, thompson_path: Optional[Path] = None) -> int:
        """Migrate current Thompson distributions from flat file to Neo4j.

        Creates Source nodes with the current alpha/beta as a single
        THOMPSON_UPDATE snapshot. Does NOT reconstruct the full update
        history — that lives in ``thompson_history.jsonl`` and is too
        large for a one-shot migration without streaming.

        Returns:
            Number of sources successfully written.
        """
        if not self.is_connected():
            return 0
        if thompson_path is None:
            thompson_path = (
                Path(__file__).resolve().parents[3]
                / "data" / "market" / "thompson_distributions.json"
            )
        if not thompson_path.exists():
            logger.debug("KnowledgeGraph: Thompson seed file not found at %s", thompson_path)
            return 0
        try:
            data = json.loads(thompson_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("KnowledgeGraph: could not read Thompson file", exc_info=True)
            return 0

        count = 0
        ts_now = datetime.now()
        for source_key, regime_data in data.items():
            for regime, params in regime_data.items():
                try:
                    self.store_thompson_update(
                        source_key=source_key,
                        regime=regime,
                        alpha=params.get("alpha", 1.0),
                        beta=params.get("beta", 1.0),
                        won=True,   # Snapshot — no outcome direction known
                        timestamp=ts_now,
                    )
                    count += 1
                except Exception:
                    logger.debug(
                        "KnowledgeGraph: Thompson source %s seed failed", source_key, exc_info=True
                    )

        logger.info("KnowledgeGraph: seeded %d Thompson source snapshots", count)
        return count
