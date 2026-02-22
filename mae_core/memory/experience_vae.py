"""Experience VAE - Variational Autoencoder for RL experience compression.

Compresses (state, action, reward, next_state) tuples into compact
latent codes (~64 dims) for generative replay. Enables 100x memory
reduction and synthetic experience generation.

Architecture:
    Encoder: (s, a, r, s') -> mu, log_var
    Decoder: z -> (s', a', r', s'_next)
    Loss: reconstruction + beta * KL divergence

Based on: Kingma & Welling (2014), Shin et al. (2017)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .experience import Experience

logger = logging.getLogger(__name__)


@dataclass
class VAEConfig:
    """Configuration for ExperienceVAE."""

    state_dim: int
    action_dim: int
    latent_dim: int = 64
    hidden_dims: list[int] | None = None
    beta: float = 1.0  # KL weight (beta-VAE)
    learning_rate: float = 1e-3
    dropout: float = 0.1
    discrete_actions: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class _Encoder(nn.Module):
    """Maps experience vector to latent distribution (mu, log_var)."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        action_size = config.action_dim
        input_dim = config.state_dim + action_size + 1 + config.state_dim

        layers: list[nn.Module] = []
        prev = input_dim
        for h in config.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(config.dropout)]
            prev = h

        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, config.latent_dim)
        self.fc_log_var = nn.Linear(prev, config.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_log_var(h)


class _Decoder(nn.Module):
    """Maps latent code to reconstructed experience components."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.discrete_actions = config.discrete_actions

        layers: list[nn.Module] = []
        prev = config.latent_dim
        for h in reversed(config.hidden_dims):
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h

        self.net = nn.Sequential(*layers)
        self.state_head = nn.Linear(prev, config.state_dim)
        self.next_state_head = nn.Linear(prev, config.state_dim)
        self.reward_head = nn.Linear(prev, 1)
        self.action_head = nn.Linear(prev, config.action_dim)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(z)
        action = self.action_head(h)
        if self.discrete_actions:
            action = F.softmax(action, dim=-1)
        return {
            "state": self.state_head(h),
            "action": action,
            "reward": self.reward_head(h),
            "next_state": self.next_state_head(h),
        }


class ExperienceVAE(nn.Module):
    """VAE for compressing and generating RL experiences.

    Args:
        config: VAEConfig with architecture parameters.
    """

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _Encoder(config)
        self.decoder = _Decoder(config)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

        self.total_train_steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # -- core API --

    def forward(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self._reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def train_on_experiences(
        self,
        experiences: list[Experience],
        epochs: int = 10,
        batch_size: int = 128,
    ) -> dict[str, list[float]]:
        """Train VAE on a batch of experiences. Returns loss history."""
        x = self._experiences_to_tensor(experiences)
        n = x.size(0)
        history: dict[str, list[float]] = {
            "total": [], "reconstruction": [], "kl": []
        }

        self.train()
        for _epoch in range(epochs):
            epoch_losses: list[dict[str, float]] = []
            for i in range(0, n, batch_size):
                batch = x[torch.randperm(n)[:batch_size]]
                recon, mu, log_var = self(batch)
                loss, ld = self._compute_loss(batch, recon, mu, log_var)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(ld)
                self.total_train_steps += 1

            for key in history:
                history[key].append(
                    float(np.mean([d[key] for d in epoch_losses]))
                )

        return history

    def encode(self, experiences: list[Experience]) -> np.ndarray:
        """Encode experiences to latent codes (deterministic = mean)."""
        self.eval()
        with torch.no_grad():
            x = self._experiences_to_tensor(experiences)
            mu, _ = self.encoder(x)
            return mu.cpu().numpy()

    def generate(
        self, num_samples: int = 1, z: np.ndarray | None = None
    ) -> list[dict[str, Any]]:
        """Generate synthetic experiences from latent space."""
        self.eval()
        with torch.no_grad():
            if z is None:
                z = np.random.randn(num_samples, self.config.latent_dim)
            z_t = torch.FloatTensor(z).to(self.device)
            recon = self.decoder(z_t)
            return self._tensor_to_dicts(recon)

    def reconstruction_error(self, experiences: list[Experience]) -> dict[str, float]:
        """Compute reconstruction error on a set of experiences."""
        self.eval()
        with torch.no_grad():
            x = self._experiences_to_tensor(experiences)
            recon, mu, log_var = self(x)
            _, ld = self._compute_loss(x, recon, mu, log_var)
            return ld

    def model_size_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "train_steps": self.total_train_steps,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_train_steps = ckpt["train_steps"]

    # -- internals --

    @staticmethod
    def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def _compute_loss(
        self,
        x: torch.Tensor,
        recon: dict[str, torch.Tensor],
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        bs = x.size(0)
        sd = self.config.state_dim
        ad = self.config.action_dim

        state = x[:, :sd]
        action = x[:, sd : sd + ad]
        reward = x[:, sd + ad : sd + ad + 1]
        next_s = x[:, sd + ad + 1 :]

        r_state = F.mse_loss(recon["state"], state, reduction="sum") / bs
        r_next = F.mse_loss(recon["next_state"], next_s, reduction="sum") / bs
        r_reward = F.mse_loss(recon["reward"], reward, reduction="sum") / bs

        if self.config.discrete_actions:
            r_action = F.cross_entropy(
                recon["action"], torch.argmax(action, dim=-1), reduction="sum"
            ) / bs
        else:
            r_action = F.mse_loss(recon["action"], action, reduction="sum") / bs

        recon_loss = r_state + r_next + r_reward + r_action
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / bs
        total = recon_loss + self.config.beta * kl

        return total, {
            "total": total.item(),
            "reconstruction": recon_loss.item(),
            "kl": kl.item(),
        }

    def _experiences_to_tensor(self, experiences: list[Experience]) -> torch.Tensor:
        sd = self.config.state_dim
        ad = self.config.action_dim
        rows = []
        for exp in experiences:
            s = self._pad(exp.state.flatten(), sd)
            if self.config.discrete_actions:
                a = np.zeros(ad)
                try:
                    idx = int(exp.action) if np.isscalar(exp.action) else int(exp.action[0])
                except (ValueError, TypeError, KeyError):
                    idx = hash(str(exp.action)) % ad
                a[idx % ad] = 1.0
            else:
                a = self._pad(np.asarray(exp.action).flatten(), ad)
            r = np.array([exp.reward])
            ns = self._pad(exp.next_state.flatten(), sd)
            rows.append(np.concatenate([s, a, r, ns]))
        return torch.FloatTensor(np.array(rows)).to(self.device)

    def _tensor_to_dicts(self, recon: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        out = []
        bs = recon["state"].size(0)
        for i in range(bs):
            s = recon["state"][i].cpu().numpy()
            a_raw = recon["action"][i].cpu().numpy()
            r = float(recon["reward"][i].cpu().numpy()[0])
            ns = recon["next_state"][i].cpu().numpy()
            a = int(np.argmax(a_raw)) if self.config.discrete_actions else a_raw
            out.append({"state": s, "action": a, "reward": r, "next_state": ns, "done": False})
        return out

    @staticmethod
    def _pad(arr: np.ndarray, length: int) -> np.ndarray:
        arr = arr[:length]
        if len(arr) < length:
            arr = np.pad(arr, (0, length - len(arr)))
        return arr

    def __repr__(self) -> str:
        mb = self.model_size_bytes() / (1024 * 1024)
        return (
            f"ExperienceVAE(state={self.config.state_dim}, "
            f"action={self.config.action_dim}, latent={self.config.latent_dim}, "
            f"size={mb:.1f}MB, device={self.device})"
        )
