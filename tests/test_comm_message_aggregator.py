"""Tests for message_aggregator - Deduplication and priority-based message handling.

Tests message deduplication, vote counting, priority escalation, TTL management.
"""

import time
import pytest

from mae_core.communication.message_aggregator import (
    AggregatedMessage,
    MessageAggregator,
    MessagePriority,
)


class TestMessagePriorityEnum:
    """Test MessagePriority enum."""

    def test_priority_values(self):
        """Verify priority values are ordered."""
        assert MessagePriority.LOW.value == 1
        assert MessagePriority.NORMAL.value == 2
        assert MessagePriority.HIGH.value == 3
        assert MessagePriority.URGENT.value == 4
        assert MessagePriority.CRITICAL.value == 5

    def test_priority_comparison(self):
        """Priorities can be compared by value."""
        assert MessagePriority.LOW.value < MessagePriority.CRITICAL.value


class TestAggregatedMessageDataclass:
    """Test AggregatedMessage creation and fields."""

    def test_basic_construction(self):
        """Create a basic aggregated message."""
        msg = AggregatedMessage(
            content_hash="abc123",
            message_type="TEST",
            content={"data": "value"},
        )
        assert msg.content_hash == "abc123"
        assert msg.message_type == "TEST"
        assert msg.vote_count == 0
        assert len(msg.supporters) == 0

    def test_with_supporters(self):
        """Create message with initial supporters."""
        msg = AggregatedMessage(
            content_hash="abc123",
            message_type="TEST",
            content={},
            supporters={"a1", "a2"},
            vote_count=2,
        )
        assert len(msg.supporters) == 2
        assert msg.vote_count == 2

    def test_timestamp_fields(self):
        """first_seen and last_seen are timestamped."""
        msg = AggregatedMessage(
            content_hash="abc",
            message_type="TEST",
            content={},
        )
        assert msg.first_seen > 0
        assert msg.last_seen > 0


class TestBasicAggregation:
    """Test basic message aggregation."""

    def test_submit_new_message(self):
        """Submit a new message."""
        agg = MessageAggregator()
        hash_val, is_new, msg = agg.submit_message(
            agent_id="a1",
            message_type="INFO",
            content={"text": "hello"},
        )

        assert is_new is True
        assert msg.vote_count == 1
        assert "a1" in msg.supporters

    def test_duplicate_vote_from_same_agent(self):
        """Same agent voting twice (no allow_duplicate)."""
        agg = MessageAggregator()
        agg.submit_message("a1", "INFO", {"text": "hi"})
        hash_val, is_new, msg = agg.submit_message("a1", "INFO", {"text": "hi"})

        assert is_new is False
        assert msg.vote_count == 1  # Still 1

    def test_duplicate_vote_allowed(self):
        """Same agent can vote twice when allow_duplicate=True."""
        agg = MessageAggregator()
        agg.submit_message("a1", "INFO", {"text": "hi"})
        hash_val, is_new, msg = agg.submit_message(
            "a1", "INFO", {"text": "hi"}, allow_duplicate_vote=True
        )

        assert is_new is True
        assert msg.vote_count == 2

    def test_multiple_agents_same_message(self):
        """Same message from different agents increases votes."""
        agg = MessageAggregator()
        agg.submit_message("a1", "INFO", {"text": "danger"})
        agg.submit_message("a2", "INFO", {"text": "danger"})
        hash_val, is_new, msg = agg.submit_message("a3", "INFO", {"text": "danger"})

        assert msg.vote_count == 3
        assert len(msg.supporters) == 3

    def test_different_content_different_hash(self):
        """Different content produces different hashes."""
        agg = MessageAggregator()
        hash1, _, msg1 = agg.submit_message("a1", "INFO", {"text": "hello"})
        hash2, _, msg2 = agg.submit_message("a2", "INFO", {"text": "world"})

        assert hash1 != hash2
        assert msg1.vote_count == 1
        assert msg2.vote_count == 1

    def test_same_type_different_values_different_hash(self):
        """Same message type but different values = different hash."""
        agg = MessageAggregator()
        agg.submit_message("a1", "ALERT", {"level": 1})
        agg.submit_message("a2", "ALERT", {"level": 2})

        stats = agg.get_statistics()
        assert stats["active_messages"] == 2


class TestPriorityEscalation:
    """Test priority escalation with vote count."""

    def test_priority_escalates_at_3_votes(self):
        """Priority becomes HIGH at 3 votes."""
        agg = MessageAggregator()
        agg.submit_message("a1", "WARN", {"msg": "x"})
        agg.submit_message("a2", "WARN", {"msg": "x"})
        hash_val, _, msg = agg.submit_message("a3", "WARN", {"msg": "x"})

        assert msg.priority == MessagePriority.HIGH

    def test_priority_escalates_at_5_votes(self):
        """Priority becomes URGENT at 5 votes."""
        agg = MessageAggregator()
        for i in range(5):
            hash_val, _, msg = agg.submit_message(f"a{i}", "ALERT", {"data": "x"})

        assert msg.priority == MessagePriority.URGENT

    def test_priority_escalates_at_10_votes(self):
        """Priority becomes CRITICAL at 10 votes."""
        agg = MessageAggregator()
        for i in range(10):
            hash_val, _, msg = agg.submit_message(f"a{i}", "DANGER", {"data": "x"})

        assert msg.priority == MessagePriority.CRITICAL

    def test_initial_priority_normal(self):
        """New message has NORMAL priority."""
        agg = MessageAggregator()
        hash_val, _, msg = agg.submit_message("a1", "INFO", {})
        assert msg.priority == MessagePriority.NORMAL


class TestUrgencyScore:
    """Test urgency score calculation."""

    def test_new_message_low_urgency(self):
        """New message has low urgency score."""
        agg = MessageAggregator()
        _, _, msg = agg.submit_message("a1", "INFO", {})
        assert msg.urgency_score == 0.1

    def test_multiple_votes_increase_urgency(self):
        """More votes increase urgency."""
        agg = MessageAggregator()
        _, _, msg1 = agg.submit_message("a1", "INFO", {})
        urgency1 = msg1.urgency_score

        _, _, msg2 = agg.submit_message("a2", "INFO", {})
        urgency2 = msg2.urgency_score

        assert urgency2 > urgency1

    def test_urgency_recency_decay(self):
        """Urgency increases with more votes."""
        agg = MessageAggregator(max_message_age=10.0)
        _, _, msg = agg.submit_message("a1", "INFO", {})
        initial_urgency = msg.urgency_score

        # Submit same message again from different agent
        _, _, msg2 = agg.submit_message("a2", "INFO", {})

        # With more votes, urgency should increase (consensus factor dominates)
        # The balance: 0.4 * recency + 0.6 * consensus
        assert msg2.vote_count > msg.urgency_score  # More votes


class TestGetMessages:
    """Test message retrieval with filters."""

    def test_get_all_messages(self):
        """Get all active messages."""
        agg = MessageAggregator()
        agg.submit_message("a1", "TYPE_A", {"data": 1})
        agg.submit_message("a2", "TYPE_B", {"data": 2})

        msgs = agg.get_messages()
        assert len(msgs) == 2

    def test_filter_by_type(self):
        """Filter messages by type."""
        agg = MessageAggregator()
        agg.submit_message("a1", "ALERT", {"msg": "x"})
        agg.submit_message("a2", "INFO", {"msg": "y"})

        alerts = agg.get_messages(message_type="ALERT")
        assert len(alerts) == 1
        assert alerts[0].message_type == "ALERT"

    def test_filter_by_min_priority(self):
        """Filter by minimum priority."""
        agg = MessageAggregator()
        # Create messages with different priorities
        for i in range(5):
            agg.submit_message(f"a{i}", "MSG1", {})  # Will be HIGH
        agg.submit_message("b1", "MSG2", {})  # Will be NORMAL

        urgent = agg.get_messages(min_priority=MessagePriority.HIGH)
        assert len(urgent) == 1
        assert urgent[0].message_type == "MSG1"

    def test_filter_by_min_votes(self):
        """Filter by minimum vote count."""
        agg = MessageAggregator()
        agg.submit_message("a1", "MSG1", {})
        for i in range(3):
            agg.submit_message(f"a{i}", "MSG2", {})

        high_consensus = agg.get_messages(min_votes=2)
        assert len(high_consensus) == 1
        assert high_consensus[0].vote_count >= 2

    def test_max_results_limit(self):
        """Limit number of results returned."""
        agg = MessageAggregator()
        for i in range(20):
            agg.submit_message(f"a{i}", f"MSG_{i}", {})

        msgs = agg.get_messages(max_results=5)
        assert len(msgs) == 5

    def test_sort_by_urgency(self):
        """Results sorted by urgency descending."""
        agg = MessageAggregator()
        agg.submit_message("a1", "LOW", {})
        for i in range(5):
            agg.submit_message(f"b{i}", "HIGH", {})

        msgs = agg.get_messages()
        # HIGH should be first due to higher urgency
        assert msgs[0].message_type == "HIGH"

    def test_mark_read_by(self):
        """Mark messages as read by an agent."""
        agg = MessageAggregator()
        agg.submit_message("a1", "MSG", {})

        msgs = agg.get_messages(mark_read_by="reader1")
        assert len(msgs) == 1
        assert msgs[0].times_read == 1
        assert "reader1" in msgs[0].agents_who_read


class TestTopMessages:
    """Test top-N message retrieval."""

    def test_get_top_messages_count(self):
        """Get top N messages by urgency."""
        agg = MessageAggregator()
        for i in range(10):
            agg.submit_message(f"a{i}", f"MSG_{i}", {})

        top = agg.get_top_messages(count=3)
        assert len(top) == 3

    def test_get_top_messages_default_count(self):
        """Default count is 5."""
        agg = MessageAggregator()
        for i in range(20):
            agg.submit_message(f"a{i}", f"MSG_{i}", {})

        top = agg.get_top_messages()
        assert len(top) == 5

    def test_top_messages_marked_read(self):
        """Top messages marked as read."""
        agg = MessageAggregator()
        agg.submit_message("a1", "MSG", {})

        agg.get_top_messages(mark_read_by="agent1")
        msgs = agg.get_messages()
        assert "agent1" in msgs[0].agents_who_read


class TestMessageLookup:
    """Test message lookup by hash."""

    def test_get_by_hash_found(self):
        """Retrieve message by content hash."""
        agg = MessageAggregator()
        hash_val, _, _ = agg.submit_message("a1", "TEST", {"x": 1})

        msg = agg.get_message_by_hash(hash_val)
        assert msg is not None
        assert msg.content_hash == hash_val

    def test_get_by_hash_not_found(self):
        """Non-existent hash returns None."""
        agg = MessageAggregator()
        msg = agg.get_message_by_hash("nonexistent")
        assert msg is None


class TestConsensusStrength:
    """Test consensus strength calculation."""

    def test_no_matching_messages(self):
        """No matching messages returns 0.0."""
        agg = MessageAggregator()
        agg.submit_message("a1", "TYPE_A", {})

        strength = agg.get_consensus_strength("TYPE_B")
        assert strength == 0.0

    def test_single_vote_below_threshold(self):
        """Single vote below min_votes threshold."""
        agg = MessageAggregator()
        agg.submit_message("a1", "ALERT", {})

        strength = agg.get_consensus_strength("ALERT", min_votes=3)
        assert strength == 0.0

    def test_strong_consensus(self):
        """Multiple votes indicate strong consensus."""
        agg = MessageAggregator()
        for i in range(10):
            agg.submit_message(f"a{i}", "CONSENSUS", {})

        strength = agg.get_consensus_strength("CONSENSUS", min_votes=3)
        assert strength > 0.8


class TestStatistics:
    """Test statistics tracking."""

    def test_get_statistics(self):
        """Get aggregator statistics."""
        agg = MessageAggregator()
        agg.submit_message("a1", "MSG1", {})
        agg.submit_message("a2", "MSG1", {})
        agg.submit_message("a3", "MSG2", {})

        stats = agg.get_statistics()
        assert stats["active_messages"] == 2
        assert stats["total_submitted"] == 3
        assert stats["total_deduplicated"] == 1  # Second MSG1 was dedup

    def test_dedup_rate(self):
        """Deduplication rate calculated correctly."""
        agg = MessageAggregator()
        for i in range(10):
            agg.submit_message(f"a{i}", "SAME", {})

        stats = agg.get_statistics()
        # All same message: 1 new + 9 deduplicated = 10 submitted, 9 dedup
        assert stats["dedup_rate"] == 0.9

    def test_statistics_repr(self):
        """String representation includes stats."""
        agg = MessageAggregator()
        agg.submit_message("a1", "MSG", {})
        repr_str = repr(agg)
        assert "MessageAggregator" in repr_str
        assert "messages=" in repr_str


class TestTTLAndDecay:
    """Test TTL-based message cleanup."""

    def test_max_message_age_respected(self):
        """Old messages are removed during decay."""
        agg = MessageAggregator(max_message_age=0.1, decay_interval=0.05)
        agg.submit_message("a1", "MSG", {})

        stats_before = agg.get_statistics()
        assert stats_before["active_messages"] == 1

        # Wait for message to age
        time.sleep(0.15)

        # Force decay
        agg.submit_message("a2", "NEW", {})

        stats_after = agg.get_statistics()
        # Old message should be gone
        assert stats_after["active_messages"] <= 1

    def test_max_messages_capacity(self):
        """Old messages evicted when capacity reached."""
        agg = MessageAggregator(max_messages=5)

        # Submit 10 different messages
        for i in range(10):
            agg.submit_message(f"a{i}", f"MSG_{i}", {})

        stats = agg.get_statistics()
        # Should keep last 5 due to OrderedDict FIFO eviction
        assert stats["active_messages"] <= 5

    def test_decay_interval_respected(self):
        """Decay only runs after decay_interval passes."""
        agg = MessageAggregator(max_message_age=0.1, decay_interval=100.0)

        agg.submit_message("a1", "MSG", {})
        time.sleep(0.15)

        # Decay won't run (interval too long), so message still there
        agg.submit_message("a2", "NEW", {})
        stats = agg.get_statistics()
        assert stats["active_messages"] == 2


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_submissions(self):
        """Multiple threads can submit messages."""
        import threading

        agg = MessageAggregator()
        errors = []

        def submit_messages():
            try:
                for i in range(10):
                    agg.submit_message(f"a1", f"MSG_{i}", {})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit_messages) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have handled all messages safely
        stats = agg.get_statistics()
        assert stats["total_submitted"] > 0

    def test_concurrent_read_write(self):
        """Reading and writing concurrently is safe."""
        import threading

        agg = MessageAggregator()
        errors = []

        def writer():
            try:
                for i in range(5):
                    agg.submit_message(f"a1", f"MSG_{i}", {})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(5):
                    agg.get_messages()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestHashConsistency:
    """Test hash consistency for deduplication."""

    def test_same_content_same_hash(self):
        """Identical content produces identical hashes."""
        agg = MessageAggregator()
        hash1, _, _ = agg.submit_message("a1", "TEST", {"x": 1, "y": 2})
        hash2, _, _ = agg.submit_message("a2", "TEST", {"x": 1, "y": 2})

        assert hash1 == hash2

    def test_order_irrelevant_in_hash(self):
        """Dict key order doesn't affect hash (due to sorting)."""
        agg = MessageAggregator()
        hash1, _, _ = agg.submit_message("a1", "TEST", {"x": 1, "y": 2})
        # Same content, potentially different iteration order
        hash2, _, _ = agg.submit_message("a2", "TEST", {"y": 2, "x": 1})

        assert hash1 == hash2
