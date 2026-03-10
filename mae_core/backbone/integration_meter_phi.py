"""Integration Meter — phi computation functions.

Extracted from integration_meter.py for single-responsibility.

Contains:
  _compute_entropy         — Shannon entropy via histogram binning
  _compute_joint_entropy   — Joint entropy of multiple columns
  _enumerate_bipartitions  — All non-trivial bipartitions
  _compute_phi             — IIT Phi for a set of children
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Any, Optional

import numpy as np

from mae_core.backbone.integration_meter_models import PhiResult


def compute_entropy(data: np.ndarray, bins: int = 8) -> float:
    """Shannon entropy of a 1D array via histogram binning.

    H(X) = -sum(p * log2(p)) for each bin with p > 0.
    """
    if len(data) < 2:
        return 0.0
    counts, _ = np.histogram(data, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_joint_entropy(columns: list[np.ndarray], bins: int = 8) -> float:
    """Joint entropy of multiple columns via ND histogram.

    H(X1, X2, ..., Xn) = -sum(p * log2(p)) over all joint bins.
    Uses fewer bins for higher dimensions to avoid sparse histograms.
    """
    if not columns or len(columns[0]) < 2:
        return 0.0

    n_dims = len(columns)
    # Reduce bins for higher dimensions to avoid sparse histograms
    # 3D: use bins/2, 4D+: use bins/3 (but minimum 3)
    if n_dims <= 2:
        joint_bins = bins
    elif n_dims == 3:
        joint_bins = max(3, bins // 2)
    else:
        joint_bins = max(3, bins // 3)

    sample = np.column_stack(columns)
    try:
        counts, _ = np.histogramdd(sample, bins=joint_bins)
    except (ValueError, MemoryError):
        return 0.0

    flat = counts.flatten()
    total = flat.sum()
    if total == 0:
        return 0.0
    probs = flat / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def enumerate_bipartitions(items: list) -> list[tuple[tuple, tuple]]:
    """All non-trivial bipartitions of items.

    A bipartition splits items into two non-empty groups.
    For 3 items: 3 partitions. For 4: 7. For 5: 15.
    """
    n = len(items)
    if n < 2:
        return []

    partitions = []
    # Generate all subsets of size 1 to n//2
    for size in range(1, n // 2 + 1):
        for combo in combinations(range(n), size):
            group_a = tuple(items[i] for i in combo)
            group_b = tuple(items[i] for i in range(n) if i not in combo)
            # Avoid duplicate when size == n//2 and n is even
            if size == n // 2 and n % 2 == 0:
                # Only include if first element is in group_a
                if combo[0] != 0:
                    continue
            partitions.append((group_a, group_b))

    return partitions


def compute_phi(
    holon_id: str,
    holon_type: str,
    children_ids: list[str],
    buffers: dict[str, deque],
    bins: int = 8,
) -> Optional[PhiResult]:
    """Compute IIT Phi for a set of children using buffered state histories.

    1. Build joint state matrix (N samples x len(children))
    2. Enumerate all bipartitions
    3. For each partition: phi_i = H(part_A) + H(part_B) - H(whole)
       This is the mutual information across the cut.
    4. Phi = min(phi_i) -- the Minimum Information Partition (MIP)
    5. If Phi > 0: genuine integration (partitioning loses information)
    """
    # Check we have enough data
    valid_children = [c for c in children_ids if c in buffers and len(buffers[c]) >= 10]
    if len(valid_children) < 2:
        return None

    # Build aligned state matrix
    min_len = min(len(buffers[c]) for c in valid_children)
    columns = {c: np.array(list(buffers[c]))[-min_len:] for c in valid_children}

    # Compute whole-system joint entropy
    all_cols = [columns[c] for c in valid_children]
    h_whole = compute_joint_entropy(all_cols, bins=bins)

    # Enumerate bipartitions
    bipartitions = enumerate_bipartitions(valid_children)
    if not bipartitions:
        return None

    # Compute phi for each bipartition
    partition_results = []
    for group_a, group_b in bipartitions:
        cols_a = [columns[c] for c in group_a]
        cols_b = [columns[c] for c in group_b]

        h_a = compute_joint_entropy(cols_a, bins=bins)
        h_b = compute_joint_entropy(cols_b, bins=bins)

        # Mutual information across the cut:
        # phi = H(A) + H(B) - H(A,B)
        # Positive means partitioning loses information
        phi_i = h_a + h_b - h_whole
        partition_results.append({
            "group_a": group_a,
            "group_b": group_b,
            "phi": max(0.0, phi_i),  # Floor at 0 (numerical noise)
            "h_a": h_a,
            "h_b": h_b,
            "h_whole": h_whole,
        })

    # MIP: partition with minimum phi (weakest integration point)
    mip_result = min(partition_results, key=lambda x: x["phi"])
    phi = mip_result["phi"]
    mip = (mip_result["group_a"], mip_result["group_b"])

    return PhiResult(
        holon_id=holon_id,
        holon_type=holon_type,
        phi=phi,
        mip=mip,
        all_partitions=partition_results,
        children_ids=valid_children,
        buffer_size=min_len,
    )
