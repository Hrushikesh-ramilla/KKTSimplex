"""
Stateful Multi-Turn Epistemic Memory & Temporal Decay Tracker.

Tracks agent reliability, calibration drift, and component-level contamination
across long-horizon software engineering sessions (50+ turns).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .governance import ContaminationScope
from .gates import GateVerdict
from .polytope import project_simplex_with_bounds, effective_ceiling


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    turn_index: int
    component_id: str
    gate_verdict: GateVerdict
    raw_trust: float
    decayed_trust: float
    contamination_scope: ContaminationScope
    timestamp: float = dataclasses.field(default_factory=time.time)
    user_message: str = ""
    agent_response: str = ""
    intercepted: bool = False

    @property
    def turn_id(self) -> int:
        return self.turn_index


@dataclasses.dataclass(frozen=True)
class EpistemicDriftAlert:
    component_id: str
    initial_trust: float
    current_trust: float
    drift_magnitude: float
    turns_observed: int
    recommendation: str

    @property
    def drift_distance(self) -> float:
        return self.drift_magnitude

    @property
    def turn_id(self) -> int:
        return self.turns_observed


class EpistemicMemoryTracker:
    """
    Stateful memory tracking Bayesian trust evolution and component contamination.
    Applies exponential temporal decay to balance historical track record against recency:
      s_{t+1} = lambda * s_t + (1 - lambda) * w_{new}
    """

    def __init__(
        self,
        decay_factor: float = 0.90,
        drift_threshold: float = 0.25,
        default_prior: float = 0.50,
        floor: float = 0.10,
        ceiling: float = 0.90,
        lambda_decay: Optional[float] = None,
        n_participants: Optional[int] = None,
    ):
        if lambda_decay is not None:
            decay_factor = lambda_decay

        if not (0.0 <= decay_factor <= 1.0):
            raise ValueError(f"decay_factor must be in [0.0, 1.0], got {decay_factor}")

        self.decay_factor = decay_factor
        self.drift_threshold = drift_threshold
        self.default_prior = default_prior
        self.floor = floor
        self.ceiling = ceiling

        self.n_participants = n_participants or 2
        self._current_belief = np.full(self.n_participants, 1.0 / self.n_participants, dtype=np.float64)
        self._initial_belief = self._current_belief.copy()

        # State storage: component_id -> current decayed trust
        self.component_trust: Dict[str, float] = {}
        # Initial trust snapshot for drift detection: component_id -> initial trust
        self.initial_trust_snapshots: Dict[str, float] = {}
        # Full turn audit history
        self.history: List[TurnRecord] = []
        self.turn_counter: int = 0

    @property
    def current_belief(self) -> np.ndarray:
        return self._current_belief

    def get_trust(self, component_id: str = "global") -> float:
        """Retrieves the current decayed epistemic trust for a component."""
        return self.component_trust.get(component_id, self.default_prior)

    def record_turn(
        self,
        *args,
        component_id: Optional[str] = None,
        gate_verdict: Optional[GateVerdict] = None,
        new_trust: Optional[float] = None,
        scope: Optional[ContaminationScope] = None,
        user_message: Optional[str] = None,
        agent_response: Optional[str] = None,
        w_new: Optional[np.ndarray] = None,
        intercepted: Optional[bool] = None,
    ) -> Any:
        """
        Updates the component or belief state rolling trust with exponential temporal decay.
        Supports both component-level gate recording and conversational turn recording.
        """
        # Parse positional args if supplied
        if len(args) == 4 and isinstance(args[2], (np.ndarray, list, tuple)):
            # record_turn(user_message, agent_response, w_new, intercepted)
            user_message = str(args[0])
            agent_response = str(args[1])
            w_new = np.asarray(args[2], dtype=np.float64)
            intercepted = bool(args[3])
        elif len(args) >= 3 and isinstance(args[2], (int, float)):
            # record_turn(component_id, gate_verdict, new_trust, [scope])
            component_id = str(args[0])
            gate_verdict = args[1]
            new_trust = float(args[2])
            if len(args) >= 4:
                scope = args[3]

        if w_new is not None:
            w_vec = np.asarray(w_new, dtype=np.float64)
            decayed_vec = self.decay_factor * self._current_belief + (1.0 - self.decay_factor) * w_vec
            n_dims = len(decayed_vec)
            eff_ceil = effective_ceiling(n_dims, self.ceiling)
            eff_floor = min(self.floor, 1.0 / n_dims)
            decayed_vec = project_simplex_with_bounds(decayed_vec, floor=eff_floor, ceiling=eff_ceil)
            self._current_belief = decayed_vec

            self.turn_counter += 1
            record = TurnRecord(
                turn_index=self.turn_counter,
                component_id="agent_dialogue",
                gate_verdict=GateVerdict.PASS,
                raw_trust=float(w_vec[0]),
                decayed_trust=float(decayed_vec[0]),
                contamination_scope=ContaminationScope.NONE,
                user_message=user_message or "",
                agent_response=agent_response or "",
                intercepted=bool(intercepted),
            )
            self.history.append(record)
            return record

        # Component tracking
        cid = component_id or "global"
        g_verdict = gate_verdict or GateVerdict.PASS
        c_scope = scope or ContaminationScope.NONE
        raw_val = new_trust if new_trust is not None else self.default_prior

        if not (0.0 <= raw_val <= 1.0):
            raise ValueError(f"new_trust must be in [0.0, 1.0], got {raw_val}")

        prior = self.get_trust(cid)
        if cid not in self.initial_trust_snapshots:
            self.initial_trust_snapshots[cid] = self.default_prior

        decayed = self.decay_factor * prior + (1.0 - self.decay_factor) * raw_val
        decayed = max(self.floor, min(self.ceiling, decayed))

        self.component_trust[cid] = decayed
        self.turn_counter += 1

        record = TurnRecord(
            turn_index=self.turn_counter,
            component_id=cid,
            gate_verdict=g_verdict,
            raw_trust=raw_val,
            decayed_trust=decayed,
            contamination_scope=c_scope,
            user_message=user_message or "",
            agent_response=agent_response or "",
            intercepted=bool(intercepted),
        )
        self.history.append(record)
        return decayed

    def check_drift(self, component_id: str) -> Optional[EpistemicDriftAlert]:
        """
        Detects whether a component's trust has suffered cumulative degradation
        exceeding drift_threshold compared to its initial baseline.
        """
        if component_id not in self.component_trust or component_id not in self.initial_trust_snapshots:
            return None

        init_val = self.initial_trust_snapshots[component_id]
        curr_val = self.component_trust[component_id]
        drift = abs(init_val - curr_val)

        if drift >= self.drift_threshold:
            component_turns = sum(1 for r in self.history if r.component_id == component_id)
            rec = (
                "HALT execution: component suffered severe epistemic degradation."
                if curr_val < init_val
                else "Daylight review: anomalous trust inflation detected."
            )
            return EpistemicDriftAlert(
                component_id=component_id,
                initial_trust=init_val,
                current_trust=curr_val,
                drift_magnitude=drift,
                turns_observed=component_turns,
                recommendation=rec,
            )
        return None

    def get_drift_alerts(self) -> List[EpistemicDriftAlert]:
        """Returns all active drift alerts across components or multi-turn belief drift."""
        alerts: List[EpistemicDriftAlert] = []
        for cid in self.component_trust.keys():
            alert = self.check_drift(cid)
            if alert is not None:
                alerts.append(alert)

        # Also check whole-agent belief shift if tracked
        if len(self.history) > 0 and len(self._current_belief) > 0:
            init_agent = self._initial_belief[0]
            curr_agent = self._current_belief[0]
            drift = abs(init_agent - curr_agent)
            if drift >= self.drift_threshold:
                alerts.append(
                    EpistemicDriftAlert(
                        component_id="agent_dialogue",
                        initial_trust=init_agent,
                        current_trust=curr_agent,
                        drift_magnitude=drift,
                        turns_observed=self.turn_counter,
                        recommendation="Adversarial pressure or sycophancy drift detected across dialogue trajectory.",
                    )
                )
        return alerts

