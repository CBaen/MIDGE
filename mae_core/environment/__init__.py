"""Mae's action environment - abstract problem space for agent tasks.

Biological analogy: The external world that provides problems to solve.
Without this, agents have brains but no body.
"""

from mae_core.environment.task_pool import Task, TaskPool

__all__ = ["Task", "TaskPool"]
