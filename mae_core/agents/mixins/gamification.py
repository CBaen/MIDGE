"""Gamification Mixin - Intrinsic motivation system.

Provides leveling, achievements, and experience points that
motivate agents through progression and recognition.

Ported from v5-pivot base_agent.py lines 240-251, 1133-1264.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GamificationMixin:
    """Adds gamification (levels, XP, achievements) to agents."""

    def _init_gamification(self, agent_config: dict[str, Any]) -> None:
        """Initialize gamification attributes."""
        self.agent_level: int = 1
        self.experience_points: int = 0
        self.achievements: deque[str] = deque(maxlen=100)
        self.peer_rank: Optional[int] = None
        self.team_rank: Optional[int] = None

        self.exploration_bonus: float = agent_config.get("exploration_bonus", 0.1)
        self.novelty_threshold: float = agent_config.get("novelty_threshold", 0.8)
        self.action_history: deque[Any] = deque(maxlen=100)
        self.policies_shared_with_team: int = 0

    def record_action(self, action: Any) -> None:
        """Record action for novelty detection."""
        self.action_history.append(action)

    def update_gamification(self, reward: float) -> None:
        """Update gamification metrics (levels, XP, achievements)."""
        xp_gained = int(abs(reward) * 100)
        self.experience_points += xp_gained

        xp_required = self.agent_level * 1000
        if self.experience_points >= xp_required:
            old_level = self.agent_level
            self.agent_level += 1
            logger.info(
                "%s LEVELED UP! %d -> %d (XP: %d)",
                getattr(self, "unique_id", "?"),
                old_level,
                self.agent_level,
                self.experience_points,
            )
            self.unlock_achievement(f"Level {self.agent_level} Reached")

            if self.exploration_bonus < 0.5:
                self.exploration_bonus += 0.01

        self._check_achievements()

    def _check_achievements(self) -> None:
        """Check and unlock achievements based on milestones."""
        step_count = getattr(self, "step_count", 0)
        cumulative_reward = getattr(self, "cumulative_reward", 0.0)
        has_converged = getattr(self, "has_reached_convergence", False)
        is_satisfied = getattr(self, "is_satisfied_state", False)

        milestones = [
            (step_count >= 100, "Centurion"),
            (step_count >= 1000, "Millennium"),
            (cumulative_reward >= 100, "Apprentice"),
            (cumulative_reward >= 1000, "Master"),
            (cumulative_reward >= 10000, "Grandmaster"),
            (self.policies_shared_with_team >= 50, "Team Player"),
            (self.policies_shared_with_team >= 200, "Mentor"),
            (has_converged, "Convergence Master"),
            (is_satisfied, "Satisfied Achiever"),
        ]

        for condition, name in milestones:
            if condition and name not in self.achievements:
                self.unlock_achievement(name)

    def unlock_achievement(self, achievement_name: str) -> None:
        """Unlock an achievement and grant XP bonus."""
        if achievement_name in self.achievements:
            return

        self.achievements.append(achievement_name)
        self.experience_points += 500

        logger.info(
            "%s UNLOCKED: '%s'",
            getattr(self, "unique_id", "?"),
            achievement_name,
        )

        # Emit signal if signal bus available (cross-mixin dependency)
        signal_bus = getattr(self, "signal_bus", None)
        if signal_bus is not None:
            emit_signal = getattr(self, "emit_signal", None)
            if emit_signal is not None:
                emit_signal(
                    "ACHIEVEMENT_UNLOCKED",
                    {
                        "achievement_name": achievement_name,
                        "agent_level": self.agent_level,
                        "experience_points": self.experience_points,
                    },
                )

    def _is_novel_action(self, action: Any) -> bool:
        """Check if an action is novel (not recently taken)."""
        if not self.action_history:
            return True
        recent = list(self.action_history)[-20:]
        return action not in recent

    def compute_intrinsic_reward(self, action: Any) -> float:
        """Compute intrinsic reward based on novelty."""
        if self._is_novel_action(action):
            return self.exploration_bonus
        return 0.0

    def get_gamification_status(self) -> dict[str, Any]:
        """Get current gamification status."""
        xp_required = self.agent_level * 1000
        xp_progress = (self.experience_points % xp_required) / xp_required

        return {
            "level": self.agent_level,
            "experience_points": self.experience_points,
            "xp_to_next_level": xp_required - (self.experience_points % xp_required),
            "level_progress": xp_progress,
            "achievements": list(self.achievements),
            "achievement_count": len(self.achievements),
            "peer_rank": self.peer_rank,
            "team_rank": self.team_rank,
        }

    def _serialize_gamification(self) -> dict:
        return {
            "agent_level": self.agent_level,
            "experience_points": self.experience_points,
            "achievements": list(self.achievements),
            "policies_shared_with_team": self.policies_shared_with_team,
            "peer_rank": self.peer_rank,
            "team_rank": self.team_rank,
        }

    def _restore_gamification(self, data: dict) -> None:
        self.agent_level = data.get("agent_level", 1)
        self.experience_points = data.get("experience_points", 0)
        if "achievements" in data:
            for a in data["achievements"]:
                if a not in self.achievements:
                    self.achievements.append(a)
        self.policies_shared_with_team = data.get("policies_shared_with_team", 0)
        self.peer_rank = data.get("peer_rank")
        self.team_rank = data.get("team_rank")
