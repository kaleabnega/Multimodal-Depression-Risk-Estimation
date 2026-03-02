from __future__ import annotations

from mde.core.types import TurnInput
from mde.models.audio_encoder import AudioEncoder
from mde.models.fusion import MaskedFusionMLP
from mde.models.multimodal_encoder import MultimodalEncoder
from mde.models.text_encoder import TextEncoder
from mde.models.visual_encoder import VisualEncoder
from mde.services.pipeline import DepressionRiskPipeline
from mde.services.policy import SafetyPolicyEngine
from mde.services.response import TemplateResponseGenerator


def build_pipeline() -> DepressionRiskPipeline:
    encoder = MultimodalEncoder(
        text_encoder=TextEncoder(emb_dim=16),
        audio_encoder=AudioEncoder(emb_dim=8),
        visual_encoder=VisualEncoder(emb_dim=8),
    )
    return DepressionRiskPipeline(
        encoder=encoder,
        fusion=MaskedFusionMLP(text_dim=16, audio_dim=8, visual_dim=8),
        policy=SafetyPolicyEngine(low=0.30, high=0.70),
        responder=TemplateResponseGenerator(),
    )


def main() -> None:
    pipeline = build_pipeline()

    turn = TurnInput(
        text="I feel empty and hopeless lately, and I do not know what to do.",
        audio=[0.01, 0.02, -0.01, 0.0, 0.03, -0.02, 0.0, 0.0],
        frames=["downward gaze", "flat affect", "still face"],
    )

    output = pipeline.run_turn(turn)

    print(f"risk_score={output.fusion.risk_score:.3f}")
    print(f"policy_state={output.policy.state.value}")
    print(f"reasons={output.policy.reasons}")
    print(f"response={output.response}")


if __name__ == "__main__":
    main()
