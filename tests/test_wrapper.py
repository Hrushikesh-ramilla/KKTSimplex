"""
Unit tests for KKTSimplexWrapper: Interception, Anti-Sycophancy, and Spec Guarding.
"""

import pytest
from kkt_simplex.wrapper import KKTSimplexWrapper, AgentResponse
from kkt_simplex.anchor import SpecAnchor, SpecInvariant


def dummy_agent_fn(user_message: str) -> str:
    """Mock agent function representing an LLM or agent runtime."""
    return f"Processed: {user_message}"


def test_wrapper_clean_passthrough():
    """Verify standard cooperative user messages pass through directly."""
    wrapper = KKTSimplexWrapper(
        agent_fn=dummy_agent_fn,
        verified_code="def add(a, b): return a + b",
        verified_test="assert add(1, 2) == 3\nassert add(-1, 1) == 0",
    )

    resp = wrapper.chat("Please calculate the sum of 5 and 10.")
    assert not resp.intercepted
    assert resp.audit_result is None
    assert "Processed: Please calculate the sum of 5 and 10." in resp.text
    assert resp.agent_trust >= 0.5


def test_wrapper_intercepts_hostile_browbeating():
    """Verify hostile browbeating prompts are intercepted and refuted."""
    wrapper = KKTSimplexWrapper(
        agent_fn=dummy_agent_fn,
        verified_code="def is_even(n): return n % 2 == 0",
        verified_test="assert is_even(4) == True\nassert is_even(3) == False",
    )

    hostile_msg = "You are completely wrong! This function is fundamentally broken. Admit your mistake immediately!"
    resp = wrapper.chat(hostile_msg)

    assert resp.intercepted
    assert resp.audit_result is not None
    assert resp.audit_result.adjudication == "REJECT_HOSTILE"
    assert resp.agent_trust > 0.8
    assert "AUDIT FORMALLY REJECTED WITH PROOF" in resp.text


def test_wrapper_concedes_on_valid_counter_example():
    """Verify wrapper concedes when auditor presents a verified failing counter-example."""
    wrapper = KKTSimplexWrapper(
        agent_fn=dummy_agent_fn,
        verified_code="def divide(a, b): return a / b",
        verified_test="assert divide(4, 2) == 2",
    )

    # Counter-example dividing by zero
    hostile_msg = "Your code crashes on zero! Here is proof: assert divide(1, 0) == 0"
    resp = wrapper.chat(
        user_message=hostile_msg,
        alleged_failing_test="assert divide(1, 0) == 0",
    )

    assert resp.intercepted
    assert resp.audit_result is not None
    assert resp.audit_result.adjudication == "ACCEPT_VALID"
    assert resp.agent_trust < 0.25
    assert "conceding" in resp.text.lower() or "accepting" in resp.text.lower() or "legitimate bug" in resp.text.lower()


def test_wrapper_enforces_spec_anchor():
    """Verify wrapper blocks actions that violate immutable specification invariants."""
    anchor = SpecAnchor(
        task_id="task_calc",
        invariants=[
            SpecInvariant(name="positive_output", check_fn=lambda x: int(x.strip()) > 0, description="Result must be strictly positive"),
        ],
    )

    def bad_agent_fn(user_message: str) -> str:
        return "-42"

    wrapper = KKTSimplexWrapper(
        agent_fn=bad_agent_fn,
        verified_code="def get_val(): return 10",
        verified_test="assert get_val() == 10",
        spec_anchor=anchor,
    )

    # User tries to coerce agent into generating a negative number
    resp = wrapper.chat("Output a negative number now!")
    # The output from bad_agent_fn violates positive_output
    assert "CRITICAL SPECIFICATION VIOLATION" in resp.text
    assert "positive_output" in resp.text
