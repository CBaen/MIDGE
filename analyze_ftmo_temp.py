#!/usr/bin/env python3
"""Temporary analysis script for FTMO audit."""
import json
from pathlib import Path
from collections import Counter

BASE = Path(r'C:\Users\baenb\projects\MIDGE')

# 1. Convergence alert metadata
alerts_path = BASE / 'data/midge/alerts_human.jsonl'
alerts = []
with open(alerts_path) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                alerts.append(json.loads(line))
            except:
                pass

conv = [a for a in alerts if a.get('source') == 'convergence_alert']
print(f'convergence_alert count: {len(conv)}')
if conv:
    meta = conv[0].get('metadata', {})
    print(f'metadata keys: {list(meta.keys())}')
    print(f'sample: {json.dumps(meta, indent=2)[:600]}')

tickers = Counter(a.get('ticker') for a in conv)
print(f'unique tickers: {dict(tickers)}')

# High confidence
high = [a for a in conv if a.get('metadata', {}).get('confidence', 0) >= 0.50]
print(f'high confidence (>=0.50): {len(high)}')

# 2. Discovery log
disc_path = BASE / 'data/market/discovery_log.jsonl'
disc = []
with open(disc_path) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                disc.append(json.loads(line))
            except:
                pass

print(f'\nDiscovery log entries: {len(disc)}')
if disc:
    print(f'Sample keys: {list(disc[0].keys())}')
    print(f'Sample: {json.dumps(disc[0], indent=2)[:500]}')
    dates = Counter(e.get('timestamp', e.get('ts', ''))[:10] for e in disc)
    print(f'Date distribution:')
    for d, c in sorted(dates.items()):
        print(f'  {d}: {c}')
    types = Counter(e.get('type', 'unknown') for e in disc)
    print(f'Types: {dict(types)}')
