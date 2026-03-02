from __future__ import annotations

from mde.core.types import AgentInput, PipelineOutput, UserInput
from mde.models.multimodal_encoder import MultimodalEncoder
from mde.models.fusion import MaskedFusionMLP
from mde.services.policy import SafetyPolicyEngine
from mde.services.response import TemplateResponseGenerator


class DepressionRiskPipeline:
    """End-to-end pipeline: encode -> fuse -> policy -> response."""

    def __init__(
        self,
        encoder: MultimodalEncoder,
        fusion: MaskedFusionMLP,
        policy: SafetyPolicyEngine,
        responder: TemplateResponseGenerator,
    ) -> None:
        self.encoder = encoder
        self.fusion = fusion
        self.policy = policy
        self.responder = responder

    def run_user_input(self, user_input: UserInput) -> PipelineOutput:
        features, audio_summary, visual_summary = self.encoder.encode(user_input)
        fusion_out = self.fusion.predict(features)
        policy_out = self.policy.decide(text=user_input.text, risk_score=fusion_out.risk_score)

        response = self.responder.generate(
            AgentInput(
                user_text=user_input.text,
                risk_score=fusion_out.risk_score,
                policy_state=policy_out.state,
                audio_summary=audio_summary,
                visual_summary=visual_summary,
                visual_affect_probs=features.visual_affect_probs,
            )
        )

        return PipelineOutput(
            features=features,
            fusion=fusion_out,
            policy=policy_out,
            response=response,
        )
