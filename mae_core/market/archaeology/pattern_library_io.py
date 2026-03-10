"""pattern_library_io.py - Storage and loading for pattern fingerprints + templates.

All disk I/O extracted from PatternLibrary for size compliance.
Used internally by PatternLibrary — not intended as a public API.

Functions: load_library, load_fingerprints_from_disk,
           persist_fingerprints_dict, persist_templates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from mae_core.market.archaeology.fingerprint import MoveFingerprint, PatternTemplate

logger = logging.getLogger(__name__)


def load_library(
    library_path: Path,
    templates_path: Path,
    fingerprint_ids: set,
) -> tuple[int, dict, dict]:
    """Load templates from disk; scan fingerprint file for IDs and count only.

    Returns:
        (fingerprint_count, templates_dict, template_key_index)
    """
    fp_count = 0
    if library_path.exists():
        try:
            with open(library_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        fid = data.get("fingerprint_id", "")
                        if fid:
                            fingerprint_ids.add(fid)
                        fp_count += 1
                    except json.JSONDecodeError as e:
                        logger.debug("Skipping malformed fingerprint: %s", e)
        except OSError as e:
            logger.warning("Could not load pattern library: %s", e)

    templates: dict[str, PatternTemplate] = {}
    template_key_index: dict[str, str] = {}
    tmpl_count = 0

    if templates_path.exists():
        try:
            with open(templates_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        t = PatternTemplate.from_dict(data)
                        templates[t.template_id] = t
                        template_key_index[f"{t.direction}:{t.domain_signature}"] = t.template_id
                        tmpl_count += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug("Skipping malformed template: %s", e)
        except OSError as e:
            logger.warning("Could not load pattern templates: %s", e)

    if fp_count or tmpl_count:
        logger.info(
            "Pattern library loaded: %d fingerprints (lazy), %d templates",
            fp_count, tmpl_count,
        )

    return fp_count, templates, template_key_index


def load_fingerprints_from_disk(library_path: Path) -> dict[str, MoveFingerprint]:
    """Load all fingerprints from disk into a temporary dict.

    Called only by rebuild_templates() and update_outcome(fingerprint_id=...).
    The caller is responsible for keeping or discarding the returned dict.
    """
    fingerprints: dict[str, MoveFingerprint] = {}
    if not library_path.exists():
        return fingerprints
    try:
        with open(library_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    fp = MoveFingerprint.from_dict(data)
                    fingerprints[fp.fingerprint_id] = fp
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug("Skipping malformed fingerprint on full load: %s", e)
    except OSError as e:
        logger.warning("Could not load fingerprints from disk: %s", e)
    return fingerprints


def persist_fingerprints_dict(
    library_path: Path, fingerprints: dict[str, MoveFingerprint],
) -> None:
    """Rewrite fingerprints file atomically from a provided dict."""
    if not fingerprints:
        return  # Never overwrite with empty
    try:
        tmp = library_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for fp in fingerprints.values():
                f.write(fp.to_json() + "\n")
        try:
            tmp.replace(library_path)
        except OSError:
            with open(library_path, "w") as f:
                with open(tmp, "r") as src:
                    f.write(src.read())
            try:
                tmp.unlink()
            except OSError:
                pass
    except OSError as e:
        logger.warning("Could not persist fingerprints: %s", e)


def persist_templates(
    templates_path: Path, templates: dict[str, PatternTemplate],
) -> None:
    """Rewrite templates file atomically (write .tmp then rename).

    On Windows, rename can fail if another process holds the target file.
    Falls back to direct overwrite (still safe — we have the empty guard).
    """
    if not templates:
        return  # Never overwrite with empty — protects against crash-induced data loss
    try:
        templates_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = templates_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for t in templates.values():
                f.write(t.to_json() + "\n")
        try:
            tmp.replace(templates_path)
        except OSError:
            with open(templates_path, "w") as f:
                with open(tmp, "r") as src:
                    f.write(src.read())
            try:
                tmp.unlink()
            except OSError:
                pass
    except OSError as e:
        logger.warning("Could not persist templates: %s", e)
