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

