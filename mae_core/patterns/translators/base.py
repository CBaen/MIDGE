"""Base pattern translator protocol.

Every translator implements this protocol to convert raw EventBus
messages into PatternSignals. The PatternBus uses this interface
to register translators and route events to them.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mae_core.patterns.pattern_signal import PatternSignal


@runtime_checkable
class PatternTranslator(Protocol):
    """Protocol for translating raw events into PatternSignals.

    Translators are pure listeners -- they subscribe to EventBus channels
    and convert messages into the common PatternSignal format.

    If a message doesn't contain enough information to produce a signal,
    translate() returns None (skip).
    """

    @property
    def source_name(self) -> str:
        """Name of the source system (e.g., 'world_model', 'curiosity')."""
        ...

    @property
    def channels(self) -> list[str]:
        """EventBus channels this translator subscribes to."""
        ...

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        """Translate a raw EventBus message into a PatternSignal.

        Args:
            channel: The EventBus channel the message arrived on.
            message: The serialized message (typically JSON string).

        Returns:
            A PatternSignal if the message is translatable, None to skip.
        """
        ...
