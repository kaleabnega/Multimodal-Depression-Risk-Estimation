from __future__ import annotations

from abc import ABC, abstractmethod

from mde.core.types import (
    AgentInput,
    FusionOutput,
    ModalityFeatures,
    PipelineOutput,
    PolicyDecision,
    UserInput,
)


class Encoder(ABC):
    @abstractmethod
    def encode(self, user_input: UserInput) -> ModalityFeatures:
        raise NotImplementedError


class FusionModel(ABC):
    @abstractmethod
    def predict(self, features: ModalityFeatures) -> FusionOutput:
        raise NotImplementedError


class PolicyEngine(ABC):
    @abstractmethod
    def decide(self, text: str, risk_score: float) -> PolicyDecision:
        raise NotImplementedError


class ResponseGenerator(ABC):
    @abstractmethod
    def generate(self, data: AgentInput) -> str:
        raise NotImplementedError


class RiskPipeline(ABC):
    @abstractmethod
    def run_user_input(self, user_input: UserInput) -> PipelineOutput:
        raise NotImplementedError
