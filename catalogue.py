#!/usr/bin/env python3
"""
Mae Catalogue System - SQLite + Qdrant hybrid for complete codebase accountability.

Every file across all branches tracked. Every review accountable.
No file left untouched. No corner cut.

Usage:
    python catalogue.py seed --branch main --worktree /path/to/branch
    python catalogue.py status
    python catalogue.py progress
    python catalogue.py next [--batch N] [--branch BRANCH] [--type TYPE]
    python catalogue.py review --file-id ID --summary "..." --category "..."
    python catalogue.py search [--category CAT] [--status STATUS] [--branch BRANCH]
    python catalogue.py monoliths [--branch BRANCH]
    python catalogue.py concepts
    python catalogue.py file --id ID
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "mae_catalogue.db"

# File type detection
EXTENSION_MAP = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript-react",
    ".js": "javascript",
    ".jsx": "javascript-react",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "config",
    ".cfg": "config",
    ".env": "env",
    ".sh": "shell",
    ".bat": "batch",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".xml": "xml",
    ".dockerfile": "docker",
    ".gitignore": "gitignore",
}

MONOLITH_THRESHOLD = 500  # lines


def get_db():
    """Get database connection with WAL mode and busy timeout."""
    db = sqlite3.connect(str(DB_PATH), timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("PRAGMA foreign_keys=ON")
    db.row_factory = sqlite3.Row
    return db


def init_db():
    """Create tables if they don't exist."""
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            branch TEXT NOT NULL,
            file_type TEXT,
            line_count INTEGER,
            size_bytes INTEGER,
            is_monolith BOOLEAN DEFAULT 0,
            review_status TEXT DEFAULT 'unreviewed',
            reviewer_agent TEXT,
            review_timestamp TEXT,
            UNIQUE(file_path, branch)
        );

        CREATE TABLE IF NOT EXISTS file_reviews (
            id INTEGER PRIMARY KEY,
            file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
            summary TEXT,
            feature_category TEXT,
            biological_concept TEXT,
            design_intent TEXT,
            implementation_status TEXT,
            dependencies TEXT,
            gold_notes TEXT,
            monolith_sections TEXT,
            UNIQUE(file_id)
        );

        CREATE TABLE IF NOT EXISTS cross_branch (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL UNIQUE,
            branches TEXT NOT NULL,
            differs_between_branches BOOLEAN,
            diff_summary TEXT
        );

        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY,
            file_id INTEGER REFERENCES files(id),
            agent_id TEXT NOT NULL,
            action TEXT NOT NULL,
            notes TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT,
            description TEXT,
            source_files TEXT,
            implementation_status TEXT,
            design_documents TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_files_branch ON files(branch);
        CREATE INDEX IF NOT EXISTS idx_files_status ON files(review_status);
        CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type);
        CREATE INDEX IF NOT EXISTS idx_files_monolith ON files(is_monolith);
        CREATE INDEX IF NOT EXISTS idx_reviews_category ON file_reviews(feature_category);
        CREATE INDEX IF NOT EXISTS idx_reviews_concept ON file_reviews(biological_concept);
    """)
    db.commit()
    db.close()


def detect_file_type(file_path):
    """Detect file type from extension."""
    path = Path(file_path)
    name = path.name.lower()

    # Special files
    if name == "dockerfile":
        return "docker"
    if name == "makefile":
        return "makefile"
    if name.startswith(".env"):
        return "env"
    if name == "license":
        return "license"

    ext = path.suffix.lower()
    return EXTENSION_MAP.get(ext, ext.lstrip(".") if ext else "unknown")


def count_lines(filepath):
    """Count lines in a file, handling encoding issues."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except (OSError, IOError):
        return 0


def cmd_seed(args):
    """Seed catalogue from a branch worktree."""
    worktree = Path(args.worktree)
    branch = args.branch

    if not worktree.exists():
        print(f"Error: worktree path does not exist: {worktree}")
        sys.exit(1)

    # Use git ls-files to get tracked files
    import subprocess
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(worktree),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error running git ls-files: {result.stderr}")
        sys.exit(1)

    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    print(f"Found {len(files)} files in {branch}")

    db = get_db()
    inserted = 0
    skipped = 0

    for file_path in files:
        full_path = worktree / file_path
        file_type = detect_file_type(file_path)
        line_count = count_lines(full_path)
        try:
            size_bytes = full_path.stat().st_size
        except OSError:
            size_bytes = 0

        is_monolith = line_count >= MONOLITH_THRESHOLD

        try:
            db.execute(
                """INSERT INTO files (file_path, branch, file_type, line_count, size_bytes, is_monolith)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (file_path, branch, file_type, line_count, size_bytes, is_monolith),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1

    db.commit()

    # Update cross-branch tracking
    _update_cross_branch(db)
    db.close()

    monoliths = sum(1 for f in files if count_lines(worktree / f) >= MONOLITH_THRESHOLD)
    print(f"Seeded: {inserted} new, {skipped} already existed")
    print(f"Monoliths (>={MONOLITH_THRESHOLD} lines): {monoliths}")


def _update_cross_branch(db):
    """Update cross-branch presence tracking."""
    rows = db.execute(
        """SELECT file_path, GROUP_CONCAT(DISTINCT branch) as branches, COUNT(DISTINCT branch) as branch_count
           FROM files GROUP BY file_path HAVING branch_count > 1"""
    ).fetchall()

    for row in rows:
        branches_list = json.dumps(row["branches"].split(","))
        db.execute(
            """INSERT OR REPLACE INTO cross_branch (file_path, branches, differs_between_branches)
               VALUES (?, ?, 0)""",
            (row["file_path"], branches_list),
        )
    db.commit()


def cmd_status(args):
    """Show overall catalogue status."""
    db = get_db()

    total = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    if total == 0:
        print("Catalogue is empty. Run 'seed' first.")
        return

    print("=" * 60)
    print("MAE CATALOGUE STATUS")
    print("=" * 60)

    # Per-branch breakdown
    branches = db.execute(
        """SELECT branch, COUNT(*) as total,
                  SUM(CASE WHEN review_status = 'reviewed' THEN 1 ELSE 0 END) as reviewed,
                  SUM(CASE WHEN review_status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                  SUM(CASE WHEN review_status = 'unreviewed' THEN 1 ELSE 0 END) as unreviewed,
                  SUM(CASE WHEN is_monolith = 1 THEN 1 ELSE 0 END) as monoliths
           FROM files GROUP BY branch"""
    ).fetchall()

    for b in branches:
        pct = (b["reviewed"] / b["total"] * 100) if b["total"] > 0 else 0
        print(f"\n  Branch: {b['branch']}")
        print(f"    Total files:    {b['total']}")
        print(f"    Reviewed:       {b['reviewed']} ({pct:.1f}%)")
        print(f"    In progress:    {b['in_progress']}")
        print(f"    Unreviewed:     {b['unreviewed']}")
        print(f"    Monoliths:      {b['monoliths']}")

    # File type breakdown
    print("\n  File types:")
    types = db.execute(
        """SELECT file_type, COUNT(*) as cnt FROM files GROUP BY file_type ORDER BY cnt DESC LIMIT 10"""
    ).fetchall()
    for t in types:
        print(f"    {t['file_type']:20s} {t['cnt']}")

    # Cross-branch files
    cross = db.execute("SELECT COUNT(*) FROM cross_branch").fetchone()[0]
    print(f"\n  Files in multiple branches: {cross}")

    # Concepts discovered
    concepts = db.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    print(f"  Concepts registered: {concepts}")

    # Total lines
    total_lines = db.execute("SELECT SUM(line_count) FROM files").fetchone()[0] or 0
    print(f"\n  Total lines across all branches: {total_lines:,}")
    print("=" * 60)

    db.close()


def cmd_progress(args):
    """Show review progress as a dashboard."""
    db = get_db()

    total = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    reviewed = db.execute("SELECT COUNT(*) FROM files WHERE review_status = 'reviewed'").fetchone()[0]
    in_progress = db.execute("SELECT COUNT(*) FROM files WHERE review_status = 'in_progress'").fetchone()[0]
    unreviewed = db.execute("SELECT COUNT(*) FROM files WHERE review_status = 'unreviewed'").fetchone()[0]

    if total == 0:
        print("No files in catalogue.")
        return

    pct = reviewed / total * 100
    bar_len = 40
    filled = int(bar_len * reviewed / total)
    bar = "#" * filled + "-" * (bar_len - filled)

    print(f"\n  [{bar}] {pct:.1f}%")
    print(f"  {reviewed}/{total} files reviewed | {in_progress} in progress | {unreviewed} remaining")

    # Per-category progress
    categories = db.execute(
        """SELECT fr.feature_category, COUNT(*) as cnt
           FROM file_reviews fr
           WHERE fr.feature_category IS NOT NULL
           GROUP BY fr.feature_category ORDER BY cnt DESC"""
    ).fetchall()
    if categories:
        print("\n  Categories discovered:")
        for c in categories:
            print(f"    {c['feature_category']:25s} {c['cnt']} files")

    # Recent activity
    recent = db.execute(
        """SELECT al.agent_id, al.action, f.file_path, al.timestamp
           FROM activity_log al JOIN files f ON al.file_id = f.id
           ORDER BY al.timestamp DESC LIMIT 5"""
    ).fetchall()
    if recent:
        print("\n  Recent activity:")
        for r in recent:
            print(f"    [{r['timestamp'][:16]}] {r['agent_id']}: {r['action']} - {r['file_path']}")

    db.close()


def cmd_next(args):
    """Get next batch of unreviewed files."""
    db = get_db()

    query = "SELECT id, file_path, branch, file_type, line_count, is_monolith FROM files WHERE review_status = 'unreviewed'"
    params = []

    if args.branch:
        query += " AND branch = ?"
        params.append(args.branch)
    if args.type:
        query += " AND file_type = ?"
        params.append(args.type)
    if args.monoliths_only:
        query += " AND is_monolith = 1"

    query += " ORDER BY line_count ASC LIMIT ?"
    params.append(args.batch)

    rows = db.execute(query, params).fetchall()
    if not rows:
        print("No unreviewed files matching criteria.")
        return

    print(f"Next {len(rows)} files to review:")
    for r in rows:
        mono = " [MONOLITH]" if r["is_monolith"] else ""
        print(f"  ID:{r['id']:5d} | {r['branch']:20s} | {r['line_count']:6d} lines | {r['file_type']:12s} | {r['file_path']}{mono}")

    db.close()


def cmd_review(args):
    """Record a file review."""
    db = get_db()

    file = db.execute("SELECT * FROM files WHERE id = ?", (args.file_id,)).fetchone()
    if not file:
        print(f"Error: No file with id {args.file_id}")
        sys.exit(1)

    now = datetime.utcnow().isoformat()
    agent = args.agent or "unknown"

    # Update file status
    db.execute(
        """UPDATE files SET review_status = 'reviewed', reviewer_agent = ?, review_timestamp = ?
           WHERE id = ?""",
        (agent, now, args.file_id),
    )

    # Insert or update review
    db.execute(
        """INSERT OR REPLACE INTO file_reviews
           (file_id, summary, feature_category, biological_concept, design_intent,
            implementation_status, dependencies, gold_notes, monolith_sections)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            args.file_id,
            args.summary,
            args.category,
            args.concept,
            args.intent,
            args.impl_status,
            args.dependencies,
            args.gold,
            args.sections,
        ),
    )

    # Log activity
    db.execute(
        """INSERT INTO activity_log (file_id, agent_id, action, notes)
           VALUES (?, ?, 'completed_review', ?)""",
        (args.file_id, agent, args.summary[:200] if args.summary else None),
    )

    db.commit()
    print(f"Reviewed: {file['file_path']} ({file['branch']})")
    db.close()


def cmd_claim(args):
    """Claim a file for review (mark as in_progress)."""
    db = get_db()
    agent = args.agent or "unknown"

    file = db.execute("SELECT * FROM files WHERE id = ?", (args.file_id,)).fetchone()
    if not file:
        print(f"Error: No file with id {args.file_id}")
        sys.exit(1)

    db.execute(
        "UPDATE files SET review_status = 'in_progress', reviewer_agent = ? WHERE id = ?",
        (agent, args.file_id),
    )
    db.execute(
        "INSERT INTO activity_log (file_id, agent_id, action) VALUES (?, ?, 'started_review')",
        (args.file_id, agent),
    )
    db.commit()
    print(f"Claimed: {file['file_path']} ({file['branch']})")
    db.close()


def cmd_search(args):
    """Search the catalogue."""
    db = get_db()

    query = """SELECT f.id, f.file_path, f.branch, f.file_type, f.line_count, f.review_status,
                      fr.summary, fr.feature_category, fr.biological_concept
               FROM files f LEFT JOIN file_reviews fr ON f.id = fr.file_id WHERE 1=1"""
    params = []

    if args.category:
        query += " AND fr.feature_category = ?"
        params.append(args.category)
    if args.concept:
        query += " AND fr.biological_concept LIKE ?"
        params.append(f"%{args.concept}%")
    if args.status:
        query += " AND f.review_status = ?"
        params.append(args.status)
    if args.branch:
        query += " AND f.branch = ?"
        params.append(args.branch)
    if args.keyword:
        query += " AND (f.file_path LIKE ? OR fr.summary LIKE ? OR fr.gold_notes LIKE ?)"
        params.extend([f"%{args.keyword}%"] * 3)

    query += " ORDER BY f.id LIMIT 50"
    rows = db.execute(query, params).fetchall()

    if not rows:
        print("No matching files.")
        return

    print(f"Found {len(rows)} files:")
    for r in rows:
        cat = f" [{r['feature_category']}]" if r["feature_category"] else ""
        concept = f" ({r['biological_concept']})" if r["biological_concept"] else ""
        print(f"  ID:{r['id']:5d} | {r['branch']:20s} | {r['review_status']:12s} | {r['file_path']}{cat}{concept}")
        if r["summary"]:
            print(f"          {r['summary'][:100]}")

    db.close()


def cmd_monoliths(args):
    """List all monolith files."""
    db = get_db()

    query = "SELECT id, file_path, branch, line_count, review_status FROM files WHERE is_monolith = 1"
    params = []
    if args.branch:
        query += " AND branch = ?"
        params.append(args.branch)
    query += " ORDER BY line_count DESC"

    rows = db.execute(query, params).fetchall()

    if not rows:
        print("No monolith files found.")
        return

    print(f"Monolith files (>={MONOLITH_THRESHOLD} lines): {len(rows)}")
    for r in rows:
        print(f"  ID:{r['id']:5d} | {r['line_count']:6d} lines | {r['review_status']:12s} | {r['branch']:20s} | {r['file_path']}")

    db.close()


def cmd_file(args):
    """Show full details for a specific file."""
    db = get_db()

    file = db.execute("SELECT * FROM files WHERE id = ?", (args.id,)).fetchone()
    if not file:
        print(f"No file with id {args.id}")
        return

    print(f"\n  File: {file['file_path']}")
    print(f"  Branch: {file['branch']}")
    print(f"  Type: {file['file_type']}")
    print(f"  Lines: {file['line_count']}")
    print(f"  Size: {file['size_bytes']} bytes")
    print(f"  Monolith: {'Yes' if file['is_monolith'] else 'No'}")
    print(f"  Status: {file['review_status']}")
    print(f"  Reviewer: {file['reviewer_agent'] or 'None'}")
    print(f"  Reviewed at: {file['review_timestamp'] or 'N/A'}")

    review = db.execute("SELECT * FROM file_reviews WHERE file_id = ?", (args.id,)).fetchone()
    if review:
        print(f"\n  Summary: {review['summary']}")
        print(f"  Category: {review['feature_category']}")
        print(f"  Biological concept: {review['biological_concept']}")
        print(f"  Design intent: {review['design_intent']}")
        print(f"  Implementation: {review['implementation_status']}")
        if review["gold_notes"]:
            print(f"  GOLD: {review['gold_notes']}")
        if review["dependencies"]:
            print(f"  Dependencies: {review['dependencies']}")
        if review["monolith_sections"]:
            print(f"  Sections: {review['monolith_sections']}")

    # Cross-branch presence
    cross = db.execute("SELECT * FROM cross_branch WHERE file_path = ?", (file["file_path"],)).fetchone()
    if cross:
        print(f"\n  Also in branches: {cross['branches']}")
        if cross["diff_summary"]:
            print(f"  Differences: {cross['diff_summary']}")

    db.close()


def cmd_concepts(args):
    """List registered concepts."""
    db = get_db()
    rows = db.execute("SELECT * FROM concepts ORDER BY category, name").fetchall()

    if not rows:
        print("No concepts registered yet.")
        return

    current_cat = None
    for r in rows:
        if r["category"] != current_cat:
            current_cat = r["category"]
            print(f"\n  [{current_cat or 'uncategorized'}]")
        status = f" ({r['implementation_status']})" if r["implementation_status"] else ""
        print(f"    {r['name']}{status}")
        if r["description"]:
            print(f"      {r['description'][:120]}")

    db.close()


def cmd_add_concept(args):
    """Register a new concept."""
    db = get_db()
    try:
        db.execute(
            """INSERT INTO concepts (name, category, description, source_files, implementation_status, design_documents)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (args.name, args.category, args.description, args.source_files, args.impl_status, args.documents),
        )
        db.commit()
        print(f"Registered concept: {args.name}")
    except sqlite3.IntegrityError:
        print(f"Concept '{args.name}' already exists. Use search to find it.")
    db.close()


def cmd_batch_review(args):
    """Record reviews from a JSON file (for agent bulk updates)."""
    with open(args.json_file, "r") as f:
        reviews = json.load(f)

    db = get_db()
    count = 0
    now = datetime.utcnow().isoformat()

    for review in reviews:
        file_id = review.get("file_id")
        if not file_id:
            continue

        agent = review.get("agent", "batch")

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

    db.commit()
    print(f"Batch reviewed {count} files.")
    db.close()


def main():
    parser = argparse.ArgumentParser(description="Mae Catalogue System")
    sub = parser.add_subparsers(dest="command")

    # seed
    p = sub.add_parser("seed", help="Seed catalogue from a branch worktree")
    p.add_argument("--branch", required=True, help="Branch name")
    p.add_argument("--worktree", required=True, help="Path to worktree")

    # status
    sub.add_parser("status", help="Show catalogue status")

    # progress
    sub.add_parser("progress", help="Show review progress dashboard")

    # next
    p = sub.add_parser("next", help="Get next unreviewed files")
    p.add_argument("--batch", type=int, default=20, help="Batch size")
    p.add_argument("--branch", help="Filter by branch")
    p.add_argument("--type", help="Filter by file type")
    p.add_argument("--monoliths-only", action="store_true", help="Only monoliths")

    # review
    p = sub.add_parser("review", help="Record a file review")
    p.add_argument("--file-id", type=int, required=True, help="File ID")
    p.add_argument("--summary", help="Plain-language summary")
    p.add_argument("--category", help="Feature category")
    p.add_argument("--concept", help="Biological concept")
    p.add_argument("--intent", help="Design intent")
    p.add_argument("--impl-status", help="Implementation status")
    p.add_argument("--dependencies", help="JSON list of dependencies")
    p.add_argument("--gold", help="Gold notes - anything unique/valuable")
    p.add_argument("--sections", help="JSON monolith sections")
    p.add_argument("--agent", help="Reviewer agent name")

    # claim
    p = sub.add_parser("claim", help="Claim a file for review")
    p.add_argument("--file-id", type=int, required=True, help="File ID")
    p.add_argument("--agent", help="Agent name")

    # search
    p = sub.add_parser("search", help="Search the catalogue")
    p.add_argument("--category", help="Filter by category")
    p.add_argument("--concept", help="Filter by biological concept")
    p.add_argument("--status", help="Filter by review status")
    p.add_argument("--branch", help="Filter by branch")
    p.add_argument("--keyword", help="Search in paths, summaries, gold notes")

    # monoliths
    p = sub.add_parser("monoliths", help="List monolith files")
    p.add_argument("--branch", help="Filter by branch")

    # file
    p = sub.add_parser("file", help="Show file details")
    p.add_argument("--id", type=int, required=True, help="File ID")

    # concepts
    sub.add_parser("concepts", help="List registered concepts")

    # add-concept
    p = sub.add_parser("add-concept", help="Register a new concept")
    p.add_argument("--name", required=True, help="Concept name")
    p.add_argument("--category", help="Category")
    p.add_argument("--description", help="Description")
    p.add_argument("--source-files", help="JSON list of file IDs")
    p.add_argument("--impl-status", help="Implementation status")
    p.add_argument("--documents", help="JSON list of doc file IDs")

    # batch-review
    p = sub.add_parser("batch-review", help="Batch review from JSON file")
    p.add_argument("--json-file", required=True, help="Path to JSON reviews file")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize DB
    init_db()

    commands = {
        "seed": cmd_seed,
        "status": cmd_status,
        "progress": cmd_progress,
        "next": cmd_next,
        "review": cmd_review,
        "claim": cmd_claim,
        "search": cmd_search,
        "monoliths": cmd_monoliths,
        "file": cmd_file,
        "concepts": cmd_concepts,
        "add-concept": cmd_add_concept,
        "batch-review": cmd_batch_review,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
