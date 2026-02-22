"""StateStore - In-process key-value store replacing Redis.

Provides agent state persistence, hash operations, and pattern
matching without requiring an external server.

Drop-in replacement for RedisClient's key-value interface.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StateStore:
    """In-process key-value store replacing Redis key-value operations.

    Supports:
    - Key-value get/set with optional expiration
    - Hash operations (set_hash, get_hash)
    - Pattern matching for key discovery (agent:state:*)
    - JSON persistence to disk on save/load
    """

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._store: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}
        self._hashes: dict[str, dict[str, Any]] = {}
        self._persist_path = persist_path

    def set_key_value(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
        px: Optional[int] = None,
    ) -> bool:
        """Store a value with optional expiration (ex=seconds, px=milliseconds)."""
        if isinstance(value, (dict, list)):
            self._store[key] = json.dumps(value)
        else:
            self._store[key] = value

        if ex is not None:
            self._expiry[key] = time.time() + ex
        elif px is not None:
            self._expiry[key] = time.time() + (px / 1000.0)
        elif key in self._expiry:
            del self._expiry[key]

        return True

    def get_key_value(
        self, key: str, deserialize_json: bool = True
    ) -> Optional[Any]:
        """Retrieve a value by key. Returns None if not found or expired."""
        self._evict_expired(key)

        value = self._store.get(key)
        if value is None:
            return None

        if deserialize_json and isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return value

    def delete_key(self, *keys: str) -> int:
        """Delete one or more keys. Returns count of keys deleted."""
        deleted = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                self._expiry.pop(key, None)
                deleted += 1
        return deleted

    def exists(self, *keys: str) -> int:
        """Check how many of the given keys exist."""
        self._evict_all_expired()
        return sum(1 for key in keys if key in self._store)

    def keys(self, pattern: str = "*") -> list[str]:
        """Return all keys matching the given pattern (fnmatch-style)."""
        self._evict_all_expired()
        return [k for k in self._store if fnmatch.fnmatch(k, pattern)]

    # --- Hash Operations ---

    def set_hash(self, key: str, mapping: dict[str, Any]) -> int:
        """Store a hash (field-value mapping) under a key."""
        if key not in self._hashes:
            self._hashes[key] = {}

        serialized = {}
        for field, value in mapping.items():
            if isinstance(value, (dict, list)):
                serialized[field] = json.dumps(value)
            else:
                serialized[field] = value

        self._hashes[key].update(serialized)
        return len(mapping)

    def get_hash(
        self, key: str, deserialize_json: bool = True
    ) -> dict[str, Any]:
        """Retrieve a hash by key."""
        raw = self._hashes.get(key, {})
        if not deserialize_json:
            return dict(raw)

        result = {}
        for field, value in raw.items():
            if isinstance(value, str):
                try:
                    result[field] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result[field] = value
            else:
                result[field] = value
        return result

    # --- Persistence ---

    def save(self, path: Optional[Path] = None) -> None:
        """Persist all state to a JSON file."""
        target = path or self._persist_path
        if target is None:
            return

        data = {
            "store": {},
            "hashes": {},
        }

        # Only save non-expired keys
        self._evict_all_expired()
        for key, value in self._store.items():
            try:
                json.dumps(value)
                data["store"][key] = value
            except (TypeError, ValueError):
                data["store"][key] = str(value)

        for key, mapping in self._hashes.items():
            data["hashes"][key] = dict(mapping)

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2))
        logger.info("StateStore saved to %s (%d keys)", target, len(data["store"]))

    def load(self, path: Optional[Path] = None) -> None:
        """Load state from a JSON file."""
        target = path or self._persist_path
        if target is None:
            return

        target = Path(target)
        if not target.exists():
            logger.info("No state file at %s, starting fresh", target)
            return

        data = json.loads(target.read_text())
        self._store = data.get("store", {})
        self._hashes = data.get("hashes", {})
        self._expiry.clear()
        logger.info("StateStore loaded from %s (%d keys)", target, len(self._store))

    # --- Internal ---

    def _evict_expired(self, key: str) -> None:
        """Remove a key if its expiration has passed."""
        if key in self._expiry and time.time() > self._expiry[key]:
            self._store.pop(key, None)
            del self._expiry[key]

    def _evict_all_expired(self) -> None:
        """Remove all expired keys."""
        now = time.time()
        expired = [k for k, t in self._expiry.items() if now > t]
        for key in expired:
            self._store.pop(key, None)
            del self._expiry[key]

    def ping(self) -> bool:
        """Always True (no connection)."""
        return True

    def close(self) -> None:
        """Save and clean up."""
        self.save()
        self._store.clear()
        self._hashes.clear()
        self._expiry.clear()

    def flushdb(self) -> None:
        """Clear all data (for testing)."""
        self._store.clear()
        self._hashes.clear()
        self._expiry.clear()
