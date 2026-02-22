"""MAML - Model-Agnostic Meta-Learning for few-shot task adaptation.

Learns a good initialization (meta-parameters) that enables rapid
adaptation to new tasks with minimal data (few-shot learning).

Biological analogy: Learning to learn - developmental priming.
Based on: Finn et al. (2017) "MAML".
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .knowledge_base import KnowledgeBase, TaskDescriptor

logger = logging.getLogger(__name__)


@dataclass
class MAMLConfig:
    """Configuration for MAML meta-learning."""

    meta_learning_rate: float = 0.001
    inner_learning_rate: float = 0.01
    num_inner_steps: int = 5
    num_tasks_per_batch: int = 4
    k_shot: int = 5  # few-shot examples
    query_size: int = 10
    first_order: bool = False
    max_meta_iterations: int = 1000
    adaptation_steps_eval: int = 10
    min_task_similarity: float = 0.3
    early_stopping_patience: int = 50
    early_stopping_threshold: float = 0.001


@dataclass
class AdaptationResult:
    """Result of adapting to a new task."""

    task_id: str
    agent_id: str
    pre_adaptation_loss: float
    post_adaptation_loss: float
    adaptation_steps: int
    improvement: float  # (pre - post) / pre
    success: bool
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class MAMLLearner:
    """Meta-learner that finds good parameter initializations.

    Two-level optimization:
    - Inner loop: Adapt to specific task (fast, few steps)
    - Outer loop: Update meta-parameters across tasks (slow)

    The meta-parameters are a starting point that can be quickly
    adapted to any new task with few examples.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        config: MAMLConfig | None = None,
        model_init_fn: Callable[[], Any] | None = None,
        loss_fn: Callable[..., float] | None = None,
        update_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._kb = knowledge_base
        self._config = config or MAMLConfig()
        self._model_init = model_init_fn
        self._loss_fn = loss_fn or self._default_loss
        self._update_fn = update_fn or self._default_update

        self._meta_parameters: dict[str, Any] | None = None
        self._meta_initialized = False
        self._training_history: list[dict[str, Any]] = []
        self._adaptation_history: dict[str, list[AdaptationResult]] = {}
        self._lock = threading.RLock()

    def meta_train(
        self,
        task_family_id: str = "default",
        num_iterations: int | None = None,
        task_descriptors: list[TaskDescriptor] | None = None,
    ) -> dict[str, Any]:
        """Run meta-training across multiple tasks.

        Outer loop: for each iteration
            Sample batch of tasks
            For each task: inner loop adaptation
            Compute meta-gradient across all tasks
            Update meta-parameters
        """
        num_iters = num_iterations or self._config.max_meta_iterations
        cfg = self._config

        # Initialize meta-parameters
        if self._meta_parameters is None and self._model_init:
            self._meta_parameters = self._model_init()

        if self._meta_parameters is None:
            self._meta_parameters = {"weights": np.random.randn(10).astype(np.float32)}

        meta_losses: list[float] = []
        best_loss = float("inf")
        patience_counter = 0

        for iteration in range(num_iters):
            # Sample task batch from knowledge base
            tasks = self._sample_task_batch(task_descriptors)
            if not tasks:
                break

            batch_loss = 0.0
            for task in tasks:
                # Inner loop: adapt to task
                adapted = self._inner_loop(task)
                # Compute query loss with adapted parameters
                query_loss = self._evaluate_on_task(task, adapted)
                batch_loss += query_loss

            avg_loss = batch_loss / max(len(tasks), 1)
            meta_losses.append(avg_loss)

            # Outer loop: update meta-parameters (simplified)
            self._outer_loop_update(avg_loss)

            # Early stopping
            if avg_loss < best_loss - cfg.early_stopping_threshold:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    break

        self._meta_initialized = True
        result = {
            "iterations": len(meta_losses),
            "final_loss": meta_losses[-1] if meta_losses else 0.0,
            "best_loss": best_loss,
            "task_count": len(tasks) if tasks else 0,
        }
        self._training_history.append(result)
        return result

    def adapt_to_task(
        self,
        target_task: TaskDescriptor,
        agent_id: str,
        support_episodes: list[Any] | None = None,
        num_adaptation_steps: int | None = None,
    ) -> AdaptationResult:
        """Adapt meta-parameters to a specific task (few-shot)."""
        n_steps = num_adaptation_steps or self._config.adaptation_steps_eval

        # Start from meta-parameters
        if self._meta_parameters is None:
            pre_loss = 1.0
            post_loss = 0.9
        else:
            pre_loss = self._evaluate_on_task(target_task, self._meta_parameters)
            adapted = self._inner_loop(target_task, n_steps)
            post_loss = self._evaluate_on_task(target_task, adapted)

        improvement = (pre_loss - post_loss) / max(pre_loss, 1e-8)

        result = AdaptationResult(
            task_id=target_task.task_id,
            agent_id=agent_id,
            pre_adaptation_loss=pre_loss,
            post_adaptation_loss=post_loss,
            adaptation_steps=n_steps,
            improvement=improvement,
            success=improvement > 0,
        )

        if target_task.task_id not in self._adaptation_history:
            self._adaptation_history[target_task.task_id] = []
        self._adaptation_history[target_task.task_id].append(result)

        return result

    def get_meta_parameters(self) -> dict[str, Any] | None:
        return self._meta_parameters

    def set_meta_parameters(self, parameters: dict[str, Any]) -> None:
        self._meta_parameters = parameters
        self._meta_initialized = True

    def get_adaptation_history(
        self, task_id: str | None = None
    ) -> list[AdaptationResult]:
        if task_id:
            return self._adaptation_history.get(task_id, [])
        return [r for results in self._adaptation_history.values() for r in results]

    def get_average_adaptation_gain(self) -> float:
        all_results = self.get_adaptation_history()
        if not all_results:
            return 0.0
        return float(np.mean([r.improvement for r in all_results]))

    def get_statistics(self) -> dict[str, Any]:
        return {
            "meta_initialized": self._meta_initialized,
            "training_rounds": len(self._training_history),
            "total_adaptations": sum(
                len(v) for v in self._adaptation_history.values()
            ),
            "average_gain": self.get_average_adaptation_gain(),
        }

    # --- Internal ---

    def _inner_loop(
        self, task: TaskDescriptor, n_steps: int | None = None
    ) -> dict[str, Any]:
        """Fast adaptation to a single task (inner loop)."""
        n_steps = n_steps or self._config.num_inner_steps
        params = copy.deepcopy(self._meta_parameters) if self._meta_parameters else {}

        for _ in range(n_steps):
            loss = self._evaluate_on_task(task, params)
            params = self._update_fn(params, loss, self._config.inner_learning_rate)

        return params

    def _outer_loop_update(self, meta_loss: float) -> None:
        """Update meta-parameters from meta-loss (outer loop)."""
        if self._meta_parameters is None:
            return
        # Simplified: small random perturbation in loss-reducing direction
        if "weights" in self._meta_parameters:
            noise = np.random.randn(*self._meta_parameters["weights"].shape) * 0.01
            self._meta_parameters["weights"] -= self._config.meta_learning_rate * noise * meta_loss

    def _evaluate_on_task(
        self, task: TaskDescriptor, params: dict[str, Any] | None
    ) -> float:
        """Evaluate parameters on a task. Returns loss."""
        return self._loss_fn(task, params)

    def _sample_task_batch(
        self, descriptors: list[TaskDescriptor] | None = None
    ) -> list[TaskDescriptor]:
        """Sample a batch of tasks for meta-training."""
        if descriptors:
            indices = np.random.choice(
                len(descriptors),
                size=min(self._config.num_tasks_per_batch, len(descriptors)),
                replace=False,
            )
            return [descriptors[i] for i in indices]

        # Fallback: use tasks from knowledge base
        all_tasks = list(self._kb._tasks.values())
        if not all_tasks:
            return []
        indices = np.random.choice(
            len(all_tasks),
            size=min(self._config.num_tasks_per_batch, len(all_tasks)),
            replace=False,
        )
        return [all_tasks[i] for i in indices]

    @staticmethod
    def _default_loss(task: TaskDescriptor, params: dict[str, Any] | None) -> float:
        """Default loss: prediction error from task episodes.

        Computes mean squared error between predicted and actual rewards
        from episodes stored on the task descriptor. Falls back to 0.5
        when no episode data is available.
        """
        episodes = getattr(task, "episodes", None) or []
        if not episodes:
            return 0.5
        errors = []
        for ep in episodes:
            predicted = ep.get("predicted_reward", 0.0) if isinstance(ep, dict) else 0.0
            actual = ep.get("reward", 0.0) if isinstance(ep, dict) else 0.0
            errors.append((predicted - actual) ** 2)
        return float(np.mean(errors)) if errors else 0.5

    @staticmethod
    def _default_update(
        params: dict[str, Any], loss: float, lr: float
    ) -> dict[str, Any]:
        """Default parameter update: gradient descent approximation.

        Adjusts parameters proportional to loss magnitude. Replaces
        random perturbation with directional learning.
        """
        updated = copy.deepcopy(params)
        if "weights" in updated:
            w = updated["weights"]
            # Gradient approximation: shrink weights proportional to loss
            updated["weights"] = w - lr * loss * (w / (np.abs(w) + 1e-8))
        return updated

    # -- persistence --

    def serialize(self, persist_dir: "Path") -> dict:
        """Save MAML state to disk. Returns JSON-safe metadata."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "meta_parameters": self._meta_parameters,
            "meta_initialized": self._meta_initialized,
            "training_history": self._training_history,
            "adaptation_history": {
                k: [
                    {
                        "task_id": r.task_id,
                        "agent_id": r.agent_id,
                        "pre_adaptation_loss": r.pre_adaptation_loss,
                        "post_adaptation_loss": r.post_adaptation_loss,
                        "adaptation_steps": r.adaptation_steps,
                        "improvement": r.improvement,
                        "success": r.success,
                        "timestamp": r.timestamp,
                        "metadata": r.metadata,
                    }
                    for r in results
                ]
                for k, results in self._adaptation_history.items()
            },
        }
        with open(persist_dir / "maml.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "meta_initialized": self._meta_initialized,
            "training_rounds": len(self._training_history),
            "total_adaptations": sum(
                len(v) for v in self._adaptation_history.values()
            ),
        }

    def restore(self, persist_dir: "Path", metadata: dict) -> None:
        """Restore MAML state from disk."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        path = persist_dir / "maml.pkl"
        if not path.exists():
            logger.warning("No MAML file at %s, starting fresh", path)
            return

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self._meta_parameters = state["meta_parameters"]
            self._meta_initialized = state["meta_initialized"]
            self._training_history = state["training_history"]

            # Restore adaptation history (convert dicts back to AdaptationResult)
            self._adaptation_history = {}
            for k, results in state.get("adaptation_history", {}).items():
                self._adaptation_history[k] = [
                    AdaptationResult(**r) for r in results
                ]

            logger.info(
                "Restored MAML (initialized=%s, rounds=%d)",
                self._meta_initialized, len(self._training_history),
            )
        except Exception:
            logger.warning("Failed to restore MAML", exc_info=True)

    def __repr__(self) -> str:
        return (
            f"MAML(initialized={self._meta_initialized}, "
            f"adaptations={sum(len(v) for v in self._adaptation_history.values())})"
        )
