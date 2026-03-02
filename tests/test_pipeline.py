from mde.core.types import PolicyState, UserInput
from mde.models.audio_encoder import AudioEncoder
from mde.models.fusion import MaskedFusionMLP
from mde.models.multimodal_encoder import MultimodalEncoder
from mde.models.text_encoder import TextEncoder
from mde.models.visual_encoder import VisualEncoder
from mde.services.pipeline import DepressionRiskPipeline
from mde.services.policy import SafetyPolicyEngine
from mde.services.response import TemplateResponseGenerator


def _build_pipeline() -> DepressionRiskPipeline:
    encoder = MultimodalEncoder(
        TextEncoder(load_pretrained=False),
        AudioEncoder(load_pretrained=False),
        VisualEncoder(load_pretrained=False),
    )
    return DepressionRiskPipeline(
        encoder=encoder,
        fusion=MaskedFusionMLP(text_dim=384, audio_dim=8, visual_dim=8),
        policy=SafetyPolicyEngine(),
        responder=TemplateResponseGenerator(),
    )


def test_text_only_flow_runs() -> None:
    pipeline = _build_pipeline()
    out = pipeline.run_user_input(UserInput(text="I feel sad and tired"))

    assert 0.0 <= out.fusion.risk_score <= 1.0
    assert out.fusion.modality_mask == [1, 0, 0]


def test_crisis_language_overrides_score() -> None:
    pipeline = _build_pipeline()
    out = pipeline.run_user_input(UserInput(text="I want to die"))

    assert out.policy.state == PolicyState.CRISIS_PROTOCOL
    assert "988" in out.response
