"""
raw_store_base.py - Base infrastructure for Raw Data Storage Layer.

Provides __init__, _get_conn, and close. All domain mixins inherit from
this class and call self._get_conn(domain) to obtain SQLite connections.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/market/raw")


class RawStoreBase:
    """
    Base infrastructure for persisting raw API data to SQLite databases.

    WAL mode for concurrent daemon read/write. One database file per domain
    in data/market/raw/. Upsert semantics (INSERT OR REPLACE) handle dedup.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self._base_dir = Path(base_dir) if base_dir else RAW_DATA_DIR
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._connections: Dict[str, sqlite3.Connection] = {}

    def _get_conn(self, domain: str) -> sqlite3.Connection:
        if domain not in self._connections:
            db_path = self._base_dir / f"{domain}.db"
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._connections[domain] = conn
        return self._connections[domain]

    def close(self):
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()
