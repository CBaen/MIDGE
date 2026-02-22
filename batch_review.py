#!/usr/bin/env python3
"""
Batch review helper for Mae Catalogue agents.

Instead of calling catalogue.py once per file (slow, shell quoting issues on Windows),
agents can write reviews to a JSON file and submit them all at once.

Usage:
    # Agent writes reviews to a JSON file, then:
    python batch_review.py reviews.json

JSON format:
[
    {
        "file_id": 42,
        "summary": "What this file does",
        "category": "memory",
        "concept": "hippocampal replay",
        "intent": "Design intent",
        "impl_status": "implemented",
        "gold": "Anything valuable",
        "sections": null,
        "agent": "reviewer-name"
    },
    ...
]
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "mae_catalogue.db"


def batch_review(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    db = sqlite3.connect(str(DB_PATH), timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")

    now = datetime.utcnow().isoformat()
    count = 0
    errors = 0

    for review in reviews:
        file_id = review.get("file_id")
        if not file_id:
            continue

        agent = review.get("agent", "batch")

        try:
            db.execute(
                """UPDATE files SET review_status = 'reviewed', reviewer_agent = ?, review_timestamp = ?
                   WHERE id = ?""",
                (agent, now, file_id),
            )

            db.execute(
                """INSERT OR REPLACE INTO file_reviews
                   (file_id, summary, feature_category, biological_concept, design_intent,
                    implementation_status, dependencies, gold_notes, monolith_sections)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_id,
                    review.get("summary"),
                    review.get("category"),
                    review.get("concept"),
                    review.get("intent"),
                    review.get("impl_status"),
                    review.get("dependencies"),
                    review.get("gold"),
                    review.get("sections"),
                ),
            )

            db.execute(
                "INSERT INTO activity_log (file_id, agent_id, action) VALUES (?, ?, 'batch_review')",
                (file_id, agent),
            )
            count += 1
        except Exception as e:
            print(f"Error on file_id {file_id}: {e}")
            errors += 1

    db.commit()
    db.close()
    print(f"Batch reviewed {count} files. Errors: {errors}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_review.py reviews.json")
        sys.exit(1)
    batch_review(sys.argv[1])
