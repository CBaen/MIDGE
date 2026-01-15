#!/usr/bin/env python3
"""
setup_qdrant.py - Initialize Qdrant collections for MIDGE

Creates the required collections with proper vector dimensions and indexes.
Run this once before starting MIDGE.

Usage:
    python scripts/setup_qdrant.py
    python scripts/setup_qdrant.py --reset  # WARNING: Deletes existing data
"""

import requests
import sys
from pathlib import Path

QDRANT_URL = "http://localhost:6333"

# MIDGE collections configuration
COLLECTIONS = {
    "trading_research": {
        "description": "Trading research chunks with decay metadata",
        "vector_size": 768,  # nomic-embed-text dimensions
        "distance": "Cosine"
    },
    "midge_signals": {
        "description": "Active trading signals (politician trades, contracts, technicals)",
        "vector_size": 768,
        "distance": "Cosine"
    },
    "midge_predictions": {
        "description": "Prediction history for learning",
        "vector_size": 768,
        "distance": "Cosine"
    },
    "midge_research": {
        "description": "External research from Gemini (credit assignment, patterns, etc.)",
        "vector_size": 768,
        "distance": "Cosine"
    }
}


def check_qdrant_running():
    """Check if Qdrant is accessible."""
    try:
        response = requests.get(f"{QDRANT_URL}/collections", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def collection_exists(name: str) -> bool:
    """Check if a collection already exists."""
    try:
        response = requests.get(f"{QDRANT_URL}/collections/{name}")
        return response.status_code == 200
    except:
        return False


def create_collection(name: str, config: dict, reset: bool = False) -> bool:
    """Create a Qdrant collection."""

    if collection_exists(name):
        if reset:
            print(f"  Deleting existing collection: {name}")
            requests.delete(f"{QDRANT_URL}/collections/{name}")
        else:
            print(f"  Collection exists: {name} (skipping)")
            return True

    print(f"  Creating collection: {name}")
    print(f"    - {config['description']}")
    print(f"    - Vector size: {config['vector_size']}")

    payload = {
        "vectors": {
            "size": config["vector_size"],
            "distance": config["distance"]
        }
    }

    response = requests.put(
        f"{QDRANT_URL}/collections/{name}",
        json=payload
    )

    if response.status_code in (200, 201):
        print(f"    [OK] Created successfully")
        return True
    else:
        print(f"    [FAIL] Failed: {response.text}")
        return False


def create_payload_indexes(name: str) -> bool:
    """Create payload indexes for faster filtering."""

    indexes = [
        ("signal_source", "keyword"),
        ("symbol", "keyword"),
        ("timestamp", "datetime"),
        ("confidence", "float"),
        ("decayed", "bool"),
    ]

    for field, field_type in indexes:
        try:
            response = requests.put(
                f"{QDRANT_URL}/collections/{name}/index",
                json={
                    "field_name": field,
                    "field_schema": field_type
                }
            )
            if response.status_code not in (200, 201):
                # Index might already exist, that's fine
                pass
        except:
            pass

    return True


def setup_all_collections(reset: bool = False):
    """Set up all MIDGE collections."""

    print("\n" + "="*60)
    print("MIDGE Qdrant Setup")
    print("="*60)

    # Check Qdrant is running
    print("\nChecking Qdrant connection...")
    if not check_qdrant_running():
        print("ERROR: Qdrant is not running!")
        print("\nStart Qdrant with:")
        print("  docker run -d -p 6333:6333 --name qdrant qdrant/qdrant")
        print("\nOr if container exists:")
        print("  docker start qdrant")
        return False

    print("  [OK] Qdrant is running")

    # Create collections
    print("\nCreating collections...")
    success = True
    for name, config in COLLECTIONS.items():
        if not create_collection(name, config, reset):
            success = False
        else:
            create_payload_indexes(name)

    # Summary
    print("\n" + "="*60)
    if success:
        print("Setup complete! MIDGE collections are ready.")
        print("\nCollections created:")
        for name in COLLECTIONS:
            print(f"  - {name}")
        print(f"\nQdrant URL: {QDRANT_URL}")
    else:
        print("Setup completed with errors. Check output above.")
    print("="*60 + "\n")

    return success


def main():
    reset = "--reset" in sys.argv

    if reset:
        print("\n⚠️  WARNING: --reset flag detected!")
        print("This will DELETE all existing data in MIDGE collections.")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return

    success = setup_all_collections(reset)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
