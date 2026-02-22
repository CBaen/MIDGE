"""Tests for quorum_sensor - Agent-level quorum sensing interface.

Tests sensor interface, rate limiting, confidence calculation, collective insights.
"""

import time
import pytest

from mae_core.communication.quorum_sensor import (
    QuorumSensor,
    TokenBucketRateLimiter,
    DepositRecord,
)
from mae_core.communication.quorum_space import QuorumSpace
from mae_core.communication.message_aggregator import MessageAggregator, MessagePriority
from mae_core.communication.consensus_metrics import ConsensusConfidenceCalculator


class TestTokenBucketRateLimiter:
    """Test rate limiting with token bucket."""

    def test_initial_tokens(self):
        """Limiter starts with burst_size tokens."""
        limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst_size=5)
        for _ in range(5):
            assert limiter.try_consume() is True
        assert limiter.try_consume() is False

    def test_rate_refill(self):
        """Tokens refill at configured rate."""
        limiter = TokenBucketRateLimiter(rate_per_second=10.0, burst_size=1)
        assert limiter.try_consume() is True
        assert limiter.try_consume() is False

        time.sleep(0.2)  # Should accumulate ~2 tokens

        # After refill, should be able to consume
        consumed = 0
        for _ in range(3):
            if limiter.try_consume():
                consumed += 1
            else:
                break
        assert consumed >= 1

    def test_burst_size_cap(self):
        """Tokens capped at burst_size."""
        limiter = TokenBucketRateLimiter(rate_per_second=100.0, burst_size=5)
        time.sleep(0.5)  # Long wait
        # Should still be capped at 5
        count = 0
        while limiter.try_consume():
            count += 1
        assert count == 5

    def test_single_token_consumption(self):
        """Each consumption uses exactly 1 token."""
        limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst_size=3)
        consumed = 0
        while limiter.try_consume():
            consumed += 1
        assert consumed == 3


class TestDepositRecord:
    """Test DepositRecord dataclass."""

    def test_basic_record(self):
        """Create a deposit record."""
        record = DepositRecord(
            signal_type="TEST",
            strength=0.5,
            concentration_after=0.75,
        )
        assert record.signal_type == "TEST"
        assert record.strength == 0.5

    def test_record_has_timestamp(self):
        """Records include timestamp."""
        record = DepositRecord()
        assert record.timestamp > 0


class TestQuorumSensorCreation:
    """Test QuorumSensor initialization."""

    def test_create_sensor(self):
        """Create a quorum sensor."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)
        assert sensor._agent_id == "a1"

    def test_sensor_with_message_aggregator(self):
        """Sensor initialized with aggregator."""
        space = QuorumSpace()
        agg = MessageAggregator()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space, message_aggregator=agg)
        assert sensor._aggregator is not None

    def test_sensor_with_confidence_calculator(self):
        """Sensor initialized with calculator."""
        space = QuorumSpace()
        calc = ConsensusConfidenceCalculator()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space, confidence_calculator=calc)
        assert sensor._confidence is not None

    def test_sensor_rate_limiting_config(self):
        """Sensor respects rate limit configuration."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space, max_deposits_per_minute=30)
        # Should have rate limiter at 30/60 = 0.5 per second
        assert sensor._limiter._rate == 0.5


class TestDeposit:
    """Test signal deposition."""

    def test_deposit_signal(self):
        """Deposit a signal."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        result = sensor.deposit("DANGER", strength=0.8)
        assert result is True

    def test_deposit_with_metadata(self):
        """Deposit includes metadata."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        result = sensor.deposit("ALERT", strength=0.5, metadata={"risk": "high"})
        assert result is True

    def test_deposit_tracks_count(self):
        """Deposits are counted."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        sensor.deposit("TEST")
        sensor.deposit("TEST")
        sensor.deposit("OTHER")

        stats = sensor.get_statistics()
        assert stats["total_deposits"] == 3
        assert stats["deposits_by_type"]["TEST"] == 2

    def test_deposit_respects_rate_limit(self):
        """Rate limiting prevents excessive deposits."""
        space = QuorumSpace()
        sensor = QuorumSensor(
            agent_id="a1",
            quorum_space=space,
            max_deposits_per_minute=2,  # Very limited
        )

        assert sensor.deposit("A") is True
        assert sensor.deposit("B") is True  # Second one allowed
        assert sensor.deposit("C") is False  # Rate limited

        time.sleep(0.6)  # Wait for refill at 2/60 = 0.033 per second
        result = sensor.deposit("D")
        # May or may not be allowed depending on timing
        assert isinstance(result, bool)

    def test_deposit_alias(self):
        """deposit_signal is alias for deposit."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        result = sensor.deposit_signal("TEST")
        assert result is True

    def test_rate_limit_violations_tracked(self):
        """Rate limit violations are counted."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space, max_deposits_per_minute=1)

        sensor.deposit("A")
        sensor.deposit("B")  # Violation

        stats = sensor.get_statistics()
        assert stats["rate_limit_violations"] == 1


class TestDepositsViaMessageAggregator:
    """Test deposits propagated to aggregator."""

    def test_deposit_submits_to_aggregator(self):
        """Deposits also submit to message aggregator."""
        space = QuorumSpace()
        agg = MessageAggregator()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space, message_aggregator=agg)

        sensor.deposit("ALERT", metadata={"level": 1})

        msgs = agg.get_messages()
        assert len(msgs) >= 1

    def test_multiple_agents_create_consensus(self):
        """Multiple sensors depositing same signal increases votes."""
        space = QuorumSpace()
        agg = MessageAggregator()

        sensor1 = QuorumSensor(agent_id="a1", quorum_space=space, message_aggregator=agg)
        sensor2 = QuorumSensor(agent_id="a2", quorum_space=space, message_aggregator=agg)

        sensor1.deposit("DANGER", metadata={})
        sensor2.deposit("DANGER", metadata={})

        msgs = agg.get_messages()
        danger_msg = next((m for m in msgs if m.message_type == "DANGER"), None)
        assert danger_msg is not None
        assert danger_msg.vote_count == 2


class TestSense:
    """Test sensing operations."""

    def test_sense_concentration(self):
        """Sense current concentration of a signal."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        space.deposit_signal("TEST", "a1")
        conc = sensor.sense("TEST")
        assert conc > 0.0

    def test_sense_nonexistent_signal(self):
        """Sensing non-existent signal returns zero."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        conc = sensor.sense("NONEXISTENT")
        assert conc == 0.0

    def test_sense_all_concentrations(self):
        """Sense all active signals."""
        space = QuorumSpace()
        space.deposit_signal("A", "a1")
        space.deposit_signal("B", "a2")

        sensor = QuorumSensor(agent_id="a1", quorum_space=space)
        all_concs = sensor.sense_all()

        assert "A" in all_concs
        assert "B" in all_concs

    def test_sense_tracked(self):
        """Sense operations are tracked."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        sensor.sense("A")
        sensor.sense("B")

        stats = sensor.get_statistics()
        assert stats["total_senses"] == 2

    def test_sense_history(self):
        """Sense operations recorded in history."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        sensor.sense("TEST")
        sensor.sense("TEST")

        history = sensor._sensing_history
        assert len(history) >= 2


class TestContributionStatus:
    """Test contribution tracking."""

    def test_has_contributed(self):
        """Check if agent has contributed to signal."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        assert sensor.has_contributed("TEST") is False

        sensor.deposit("TEST")
        assert sensor.has_contributed("TEST") is True

    def test_contributor_fraction(self):
        """Get fraction of agents contributing."""
        space = QuorumSpace()
        sensor1 = QuorumSensor(agent_id="a1", quorum_space=space)
        sensor2 = QuorumSensor(agent_id="a2", quorum_space=space)

        sensor1.deposit("TEST")
        sensor2.deposit("TEST")

        fraction = sensor1.get_contributor_fraction("TEST", total_agents=4)
        assert fraction == 0.5  # 2 out of 4


class TestConsensusPriority:
    """Test consensus-weighted priority calculation."""

    def test_no_signal_type_default_priority(self):
        """No signal type specified returns 0.5."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        priority = sensor.get_consensus_priority()
        assert priority == 0.5

    def test_consensus_priority_based_on_concentration(self):
        """Priority increases with concentration."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        space.deposit_signal("TEST", "a1", strength=0.2)
        priority1 = sensor.get_consensus_priority("TEST")

        space.deposit_signal("TEST", "a1", strength=0.8)
        priority2 = sensor.get_consensus_priority("TEST")

        assert priority2 > priority1

    def test_consensus_priority_based_on_contributors(self):
        """Priority increases with contributor count."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        space.deposit_signal("TEST", "a1")
        priority1 = sensor.get_consensus_priority("TEST")

        space.deposit_signal("TEST", "a2")
        space.deposit_signal("TEST", "a3")
        priority2 = sensor.get_consensus_priority("TEST")

        assert priority2 > priority1

    def test_priority_bounded(self):
        """Priority always in [0, 1]."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        for _ in range(100):
            space.deposit_signal("STRESS", "a1", strength=1.0)

        priority = sensor.get_consensus_priority("STRESS")
        assert 0.0 <= priority <= 1.0


class TestCollectiveInsights:
    """Test collective insight retrieval."""

    def test_get_collective_insights(self):
        """Get top collective insights."""
        space = QuorumSpace()
        agg = MessageAggregator()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space, message_aggregator=agg)

        # Create some consensus
        for i in range(5):
            agg.submit_message(f"a{i}", "CONSENSUS_MSG", {"data": "x"})

        insights = sensor.get_collective_insights(count=5)
        assert len(insights) >= 1

    def test_insights_marked_read(self):
        """Retrieved insights marked as read by sensor."""
        space = QuorumSpace()
        agg = MessageAggregator()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space, message_aggregator=agg)

        agg.submit_message("a1", "MSG", {})

        insights = sensor.get_collective_insights()
        if insights:
            assert "a1" in insights[0].agents_who_read

    def test_no_aggregator_no_insights(self):
        """Without aggregator, no insights available."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space, message_aggregator=None)

        insights = sensor.get_collective_insights()
        assert len(insights) == 0


class TestConsensusStrength:
    """Test consensus strength calculation."""

    def test_consensus_strength_with_aggregator(self):
        """Get consensus strength from aggregator."""
        space = QuorumSpace()
        agg = MessageAggregator()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space, message_aggregator=agg)

        for i in range(5):
            agg.submit_message(f"a{i}", "TEST", {})

        strength = sensor.get_consensus_strength("TEST")
        assert strength > 0.5

    def test_consensus_strength_fallback_to_concentration(self):
        """Without aggregator, use concentration."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        space.deposit_signal("TEST", "a1", strength=0.7)

        strength = sensor.get_consensus_strength("TEST")
        assert strength > 0.0

    def test_weak_consensus(self):
        """Single vote is weak consensus."""
        space = QuorumSpace()
        agg = MessageAggregator()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space, message_aggregator=agg)

        agg.submit_message("a1", "WEAK", {})

        strength = sensor.get_consensus_strength("WEAK", min_votes=3)
        assert strength == 0.0


class TestMessageConfidence:
    """Test statistical confidence calculation."""

    def test_get_message_confidence(self):
        """Calculate confidence for a message."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        space.deposit_signal("TEST", "a1")
        space.deposit_signal("TEST", "a2")
        space.deposit_signal("TEST", "a3")

        ci = sensor.get_message_confidence("TEST", {}, total_agents=10)
        assert ci is not None
        assert ci.supporters == 3

    def test_confidence_below_min_votes(self):
        """Low vote count returns None."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        space.deposit_signal("TEST", "a1")

        ci = sensor.get_message_confidence("TEST", {}, total_agents=10)
        assert ci is None  # Only 1 vote, min is 3

    def test_is_consensus_confident(self):
        """Check if consensus is statistically confident."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        for i in range(9):
            space.deposit_signal("CONSENSUS", f"a{i}")

        confident = sensor.is_consensus_confident("CONSENSUS", total_agents=10, min_confidence=0.3)
        assert confident is True

    def test_weak_consensus_not_confident(self):
        """Weak consensus fails confidence test."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        space.deposit_signal("WEAK", "a1")

        confident = sensor.is_consensus_confident(
            "WEAK", total_agents=100, min_confidence=0.5
        )
        assert confident is False


class TestStatistics:
    """Test statistics and history."""

    def test_get_statistics(self):
        """Get sensor statistics."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        sensor.deposit("A")
        sensor.deposit("B")
        sensor.sense("C")

        stats = sensor.get_statistics()
        assert stats["agent_id"] == "a1"
        assert stats["total_deposits"] == 2
        assert stats["total_senses"] >= 1

    def test_deposit_history_limit(self):
        """Deposit history respects limit."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space, max_history=10)

        for i in range(20):
            sensor.deposit("TEST")

        history = sensor._deposit_history
        assert len(history) <= 10

    def test_get_recent_deposits(self):
        """Retrieve recent deposits."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        sensor.deposit("A")
        sensor.deposit("B")
        sensor.deposit("A")

        recents = sensor.get_recent_deposits(limit=2)
        assert len(recents) <= 2

    def test_get_recent_deposits_by_type(self):
        """Filter recent deposits by type."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        sensor.deposit("A")
        sensor.deposit("B")
        sensor.deposit("A")

        recents_a = sensor.get_recent_deposits(signal_type="A")
        for record in recents_a:
            assert record.signal_type == "A"

    def test_repr(self):
        """String representation includes agent ID."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="agent_xyz", quorum_space=space)

        sensor.deposit("TEST")

        repr_str = repr(sensor)
        assert "agent_xyz" in repr_str
        assert "QuorumSensor" in repr_str


class TestMixinInterface:
    """Test interface expected by CollectiveConsensusMixin."""

    def test_sense_method_exists(self):
        """sense() method available."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        result = sensor.sense("TEST")
        assert isinstance(result, float)

    def test_get_collective_insights_exists(self):
        """get_collective_insights() method available."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        result = sensor.get_collective_insights()
        assert isinstance(result, list)

    def test_get_consensus_strength_exists(self):
        """get_consensus_strength() method available."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        result = sensor.get_consensus_strength("TEST")
        assert isinstance(result, float)

    def test_get_statistics_exists(self):
        """get_statistics() method available."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        result = sensor.get_statistics()
        assert isinstance(result, dict)


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_agents_for_priority(self):
        """Zero agents handled gracefully."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        space.deposit_signal("TEST", "a1")
        fraction = sensor.get_contributor_fraction("TEST", total_agents=0)
        assert fraction == 0.0

    def test_negative_total_agents_zero_out(self):
        """Negative total agents treated as zero."""
        space = QuorumSpace()
        sensor = QuorumSensor(agent_id="a1", quorum_space=space)

        fraction = sensor.get_contributor_fraction("TEST", total_agents=-5)
        assert fraction == 0.0

    def test_multiple_sensors_same_space(self):
        """Multiple sensors can share same space safely."""
        space = QuorumSpace()
        sensor1 = QuorumSensor(agent_id="a1", quorum_space=space)
        sensor2 = QuorumSensor(agent_id="a2", quorum_space=space)

        sensor1.deposit("SIGNAL_A")
        sensor2.deposit("SIGNAL_B")

        conc_a = sensor2.sense("SIGNAL_A")
        conc_b = sensor1.sense("SIGNAL_B")

        assert conc_a > 0.0
        assert conc_b > 0.0
