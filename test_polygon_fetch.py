"""Quick test of Polygon bulk fetcher."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from mae_core.market.archaeology.polygon_bulk_fetcher import PolygonBulkFetcher
import time

f = PolygonBulkFetcher()
start = time.time()

for sym in ["NVDA", "AAPL", "MSFT"]:
    history = f.get_daily_history(sym, days=2000)
    if history:
        print(f"{sym}: {len(history)} bars | {history[0].timestamp} to {history[-1].timestamp}")
    else:
        print(f"{sym}: NO DATA (check MASSIVE_API_KEY)")

elapsed = time.time() - start
stats = f.get_statistics()
print(f"\nCompleted in {elapsed:.1f}s | {stats}")
