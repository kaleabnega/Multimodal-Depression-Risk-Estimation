from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class UserInput:
    """Raw user input across modalities."""

    text: str
    audio: Optional[list[float]] = None
    frames: Optional[list[str]] = None


@dataclass
class ModalityFeatures:
    """Encoder outputs for one user input."""

    text_embedding: list[float]
    text_risk: Optional[float] = None
    audio_embedding: Optional[list[float]] = None
    audio_affect_probs: Optional[list[float]] = None
    visual_embedding: Optional[list[float]] = None
    visual_affect_probs: Optional[list[float]] = None


@dataclass
class FusionOutput:
    """Fused representation and calibrated risk score."""

    joint_embedding: list[float]
    modality_mask: list[int]
    risk_score: float


class PolicyState(str, Enum):
    NORMAL_SUPPORT = "NORMAL_SUPPORT"
    GENTLE_MONITORING = "GENTLE_MONITORING"
    HIGH_RISK_SUPPORT = "HIGH_RISK_SUPPORT"
    CRISIS_PROTOCOL = "CRISIS_PROTOCOL"


@dataclass
class PolicyDecision:
    risk_score: float
    state: PolicyState
    reasons: list[str] = field(default_factory=list)


@dataclass
class AgentInput:
    user_text: str
    risk_score: float
    policy_state: PolicyState
    audio_summary: Optional[str] = None
    visual_summary: Optional[str] = None


@dataclass
class PipelineOutput:
    features: ModalityFeatures
    fusion: FusionOutput
    policy: PolicyDecision
    response: str
