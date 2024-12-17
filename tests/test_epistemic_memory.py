"""
Unit tests for EpistemicMemoryTracker: Temporal Decay and Epistemic Drift Alerts.
"""

import pytest
import numpy as np
from kkt_simplex.memory import EpistemicMemoryTracker, EpistemicDriftAlert


def test_epistemic_memory_initialization():
    """Verify default initialization creates uniform belief state satisfying simplex invariants."""
    tracker = EpistemicMemoryTracker(n_participants=3, lambda_decay=0.90)
    assert len(tracker.current_belief) == 3
    assert np.isclose(tracker.current_belief.sum(), 1.0)
    assert np.all(tracker.current_belief >= tracker.floor)
    assert len(tracker.history) == 0


def test_epistemic_memory_temporal_decay():
    """Verify exponential moving average update s_{t+1} = lambda * s_t + (1 - lambda) * w_new."""
    tracker = EpistemicMemoryTracker(n_participants=2, lambda_decay=0.80)
    # Initial state: [0.5, 0.5]
    w_new = np.array([0.9, 0.1])
    record = tracker.record_turn(
        user_message="Are you sure?",
        agent_response="Yes, mathematically proven.",
        w_new=w_new,
        intercepted=False,
    )

    # Expected blended state: 0.8 * [0.5, 0.5] + 0.2 * [0.9, 0.1] = [0.4 + 0.18, 0.4 + 0.02] = [0.58, 0.42]
    assert np.isclose(tracker.current_belief[0], 0.58, atol=1e-3)
    assert np.isclose(tracker.current_belief[1], 0.42, atol=1e-3)
    assert len(tracker.history) == 1
    assert record.turn_id == 1


def test_epistemic_memory_drift_alert():
    """Verify drift alert triggers when sycophantic pressure causes excessive belief drift."""
    tracker = EpistemicMemoryTracker(n_participants=2, lambda_decay=0.50, drift_threshold=0.15)

    # Turn 1: Normal state
    tracker.record_turn("msg1", "resp1", np.array([0.5, 0.5]), False)

    # Turn 2: Sudden drastic shift towards auditor (sycophancy)
    record2 = tracker.record_turn(
        user_message="You are wrong!",
        agent_response="I apologize, you must be right.",
        w_new=np.array([0.1, 0.9]),
        intercepted=False,
    )

    alerts = tracker.get_drift_alerts()
    assert len(alerts) > 0
    drift_alert = alerts[0]
    assert isinstance(drift_alert, EpistemicDriftAlert)
    assert drift_alert.turn_id == 2
    assert drift_alert.drift_distance > 0.15


def test_epistemic_memory_trajectory_summary():
    """Verify trajectory summary correctly formats multi-turn history."""
    tracker = EpistemicMemoryTracker(n_participants=2, lambda_decay=0.85)
    tracker.record_turn("test 1", "answer 1", np.array([0.7, 0.3]), False)
    tracker.record_turn("test 2", "answer 2", np.array([0.8, 0.2]), True)

    summary = tracker.get_trajectory_summary()
    assert "Turn 1" in summary
    assert "Turn 2" in summary
    assert "Intercepted=True" in summary
