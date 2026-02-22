"""Generative Replay Memory - VAE-based experience compression.

Stores experiences temporarily, then compresses via gzip (small datasets)
or VAE (large datasets). Generates synthetic experiences on demand for
continual learning without catastrophic forgetting.

Based on: Shin et al. (2017) "Continual Learning with Deep Generative Replay"
"""

from __future__ import annotations

import gzip
import logging
import pickle
import random
import sys
import time
from typing import Any, Optional

import numpy as np

from .experience import Experience
from .experience_vae import ExperienceVAE, VAEConfig

logger = logging.getLogger(__name__)

# Threshold for switching from gzip to VAE compression
_VAE_THRESHOLD = 1000


class GenerativeReplayMemory:
    """Memory-efficient experience storage via generative replay.

    Experiences accumulate in a temp buffer. When threshold is reached,
    they are compressed (gzip for <1000, VAE for >=1000) and the raw
    buffer is trimmed. Sampling blends real and synthetic experiences.

    Args:
        state_dim: Flattened state dimension.
        action_dim: Number of discrete actions (or continuous dim).
        latent_dim: VAE latent space dimension.
        compression_threshold: Compress after this many experiences.
        synthetic_ratio: Fraction of synthetic in sampled batches.
        keep_recent: Keep this many recent uncompressed experiences.
        discrete_actions: Whether actions are discrete.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        compression_threshold: int = 5000,
        synthetic_ratio: float = 0.7,
        keep_recent: int = 10,
        discrete_actions: bool = True,
    ) -> None:
        self._vae_config = VAEConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            discrete_actions=discrete_actions,
        )
        self._vae = ExperienceVAE(self._vae_config)

        self._compression_threshold = compression_threshold
        self._synthetic_ratio = synthetic_ratio
        self._keep_recent = keep_recent

        # Temp buffer (raw experiences before compression)
        self._buffer: list[Experience] = []

        # Compression state
        self._compressed_bytes: bytes | None = None  # gzip archive
        self._compression_method: str | None = None  # 'gzip' | 'vae' | None
        self._is_trained = False  # Whether VAE has been trained
        self._n_compressions = 0

        # Counters
        self.total_stored = 0
        self.total_generated = 0

    # -- public API --

    def store(self, experience: Experience) -> None:
        """Add experience to temp buffer. Auto-compresses at threshold."""
        self._buffer.append(experience)
        self.total_stored += 1

        if len(self._buffer) >= self._compression_threshold:
            self.compress()

    def compress(self, force: bool = False) -> dict[str, Any]:
        """Compress the buffer. Gzip for small, VAE for large datasets."""
        if len(self._buffer) == 0:
            return {"status": "skipped", "reason": "empty_buffer"}
        if not force and len(self._buffer) < 10:
            return {"status": "skipped", "reason": "insufficient_data"}

        if len(self._buffer) < _VAE_THRESHOLD:
            return self._compress_gzip()
        return self._compress_vae()

    def sample(
        self, batch_size: int = 32, synthetic_ratio: float | None = None
    ) -> list[Experience]:
        """Sample hybrid batch (real + synthetic/decompressed).

        Returns up to batch_size experiences. With VAE compression,
        blends real and synthetic. With gzip, samples from decompressed pool.
        """
        if batch_size <= 0:
            return []

        ratio = synthetic_ratio if synthetic_ratio is not None else self._synthetic_ratio

        # No compression yet - sample from buffer
        if self._compression_method is None:
            return self._sample_from(self._buffer, batch_size)

        # Gzip - sample from buffer + decompressed
        if self._compression_method == "gzip":
            pool = list(self._buffer)
            pool.extend(self._decompress_gzip())
            return self._sample_from(pool, batch_size)

        # VAE - blend real and synthetic
        n_synthetic = int(batch_size * ratio)
        n_real = batch_size - n_synthetic

        real = self._sample_from(self._buffer, n_real)
        if n_synthetic > 0 and self._is_trained:
            synthetic = self.generate_synthetic_batch(n_synthetic)
        else:
            synthetic = []

        combined = real + synthetic
        random.shuffle(combined)
        return combined

    def generate_synthetic_batch(self, batch_size: int) -> list[Experience]:
        """Generate synthetic experiences from trained VAE."""
        if not self._is_trained:
            raise RuntimeError("VAE not trained yet. Call compress() first.")

        dicts = self._vae.generate(num_samples=batch_size)
        out: list[Experience] = []
        for d in dicts:
            exp = Experience(
                state=d["state"],
                action=d["action"],
                reward=d["reward"],
                next_state=d["next_state"],
                done=d.get("done", False),
                info={"synthetic": True},
                timestamp=time.time(),
            )
            # Basic quality filter: reject NaN/Inf
            if self._is_valid(exp):
                out.append(exp)

        self.total_generated += len(out)
        return out

    def clear(self) -> None:
        self._buffer.clear()
        self._compressed_bytes = None
        self._compression_method = None
        self._is_trained = False
        self._n_compressions = 0
        self._vae = ExperienceVAE(self._vae_config)

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def is_compressed(self) -> bool:
        return self._compression_method is not None

    # -- compression internals --

    def _compress_gzip(self) -> dict[str, Any]:
        t0 = time.time()
        serialized = pickle.dumps(self._buffer)
        self._compressed_bytes = gzip.compress(serialized, compresslevel=6)
        self._compression_method = "gzip"

        n_keep = min(self._keep_recent, len(self._buffer))
        n_deleted = len(self._buffer) - n_keep
        self._buffer = self._buffer[-n_keep:] if n_keep > 0 else []
        self._n_compressions += 1

        return {
            "status": "success",
            "method": "gzip",
            "compressed": n_deleted,
            "kept": n_keep,
            "bytes": len(self._compressed_bytes),
            "elapsed": time.time() - t0,
        }

    def _compress_vae(self) -> dict[str, Any]:
        t0 = time.time()
        history = self._vae.train_on_experiences(
            self._buffer, epochs=15, batch_size=256
        )
        self._is_trained = True
        self._compression_method = "vae"

        n_keep = min(self._keep_recent, len(self._buffer))
        n_deleted = len(self._buffer) - n_keep
        self._buffer = self._buffer[-n_keep:] if n_keep > 0 else []
        self._n_compressions += 1

        return {
            "status": "success",
            "method": "vae",
            "compressed": n_deleted,
            "kept": n_keep,
            "final_loss": history["total"][-1],
            "elapsed": time.time() - t0,
        }

    def _decompress_gzip(self) -> list[Experience]:
        if self._compressed_bytes is None:
            return []
        try:
            return pickle.loads(gzip.decompress(self._compressed_bytes))
        except Exception:
            logger.warning("Gzip decompression failed", exc_info=True)
            return []

    # -- helpers --

    @staticmethod
    def _sample_from(pool: list[Experience], n: int) -> list[Experience]:
        if not pool:
            return []
        return random.sample(pool, min(n, len(pool)))

    @staticmethod
    def _is_valid(exp: Experience) -> bool:
        for arr in (exp.state, exp.next_state):
            if isinstance(arr, np.ndarray):
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    return False
        if np.isnan(exp.reward) or np.isinf(exp.reward):
            return False
        return True

    # -- persistence --

    def serialize(self, persist_dir: "Path") -> dict:
        """Save generative replay state to disk. Returns JSON-safe metadata."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Save buffer + compressed bytes
        state = {
            "buffer": self._buffer,
            "compressed_bytes": self._compressed_bytes,
            "compression_method": self._compression_method,
            "is_trained": self._is_trained,
            "n_compressions": self._n_compressions,
            "total_stored": self.total_stored,
            "total_generated": self.total_generated,
        }
        with open(persist_dir / "generative_replay.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save VAE weights if trained
        if self._is_trained:
            self._vae.save_checkpoint(str(persist_dir / "vae_weights.pt"))

        return {
            "buffer_size": len(self._buffer),
            "compression_method": self._compression_method,
            "is_trained": self._is_trained,
            "total_stored": self.total_stored,
            "total_generated": self.total_generated,
        }

    def restore(self, persist_dir: "Path", metadata: dict) -> None:
        """Restore generative replay state from disk."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        path = persist_dir / "generative_replay.pkl"
        if not path.exists():
            logger.warning("No generative replay file at %s, starting fresh", path)
            return

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self._buffer = state["buffer"]
            self._compressed_bytes = state["compressed_bytes"]
            self._compression_method = state["compression_method"]
            self._is_trained = state["is_trained"]
            self._n_compressions = state["n_compressions"]
            self.total_stored = state["total_stored"]
            self.total_generated = state["total_generated"]

            # Restore VAE weights if trained
            vae_path = persist_dir / "vae_weights.pt"
            if self._is_trained and vae_path.exists():
                self._vae.load_checkpoint(str(vae_path))

            logger.info(
                "Restored generative replay (method=%s, buffer=%d)",
                self._compression_method, len(self._buffer),
            )
        except Exception:
            logger.warning("Failed to restore generative replay", exc_info=True)

    def __len__(self) -> int:
        return len(self._buffer) + (
            self.total_stored - len(self._buffer) if self._is_trained else 0
        )

    def __repr__(self) -> str:
        method = self._compression_method or "none"
        return (
            f"GenerativeReplayMemory(method={method}, "
            f"buffer={len(self._buffer)}, "
            f"compressions={self._n_compressions})"
        )
