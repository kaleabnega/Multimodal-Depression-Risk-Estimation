from __future__ import annotations

from mde.models.audio_encoder import AudioEncoder
from mde.models.fusion import MaskedFusionMLP
from mde.models.hf_api_audio_encoder import HFAPIAudioEncoder
from mde.models.hf_api_text_encoder import HFAPITextEncoder
from mde.models.hf_api_visual_encoder import HFAPIVisualEncoder
from mde.models.multimodal_encoder import MultimodalEncoder
from mde.models.text_encoder import TextEncoder
from mde.models.visual_encoder import VisualEncoder
from mde.services.pipeline import DepressionRiskPipeline
from mde.services.policy import SafetyPolicyEngine
from mde.services.response import GuardedLLMResponseGenerator, TemplateResponseGenerator


def build_pipeline(
    *,
    backend: str = "hf_api",
    response_backend: str = "guarded_llm",
    load_pretrained: bool = True,
    local_files_only: bool = False,
    allow_fallback: bool = False,
    api_token: str | None = None,
    response_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
) -> DepressionRiskPipeline:
    if backend == "hf_api":
        encoder = MultimodalEncoder(
            text_encoder=HFAPITextEncoder(api_token=api_token, allow_fallback=allow_fallback),
            audio_encoder=HFAPIAudioEncoder(api_token=api_token, allow_fallback=allow_fallback),
            visual_encoder=HFAPIVisualEncoder(api_token=api_token, allow_fallback=allow_fallback),
        )
        fusion = MaskedFusionMLP(text_dim=384, audio_dim=8, visual_dim=8)
    else:
        encoder = MultimodalEncoder(
            text_encoder=TextEncoder(
                load_pretrained=load_pretrained,
                local_files_only=local_files_only,
                allow_fallback=allow_fallback,
            ),
            audio_encoder=AudioEncoder(
                load_pretrained=load_pretrained,
                local_files_only=local_files_only,
                allow_fallback=allow_fallback,
            ),
            visual_encoder=VisualEncoder(
                load_pretrained=load_pretrained,
                local_files_only=local_files_only,
                allow_fallback=allow_fallback,
            ),
        )
        fusion = MaskedFusionMLP(text_dim=384, audio_dim=768, visual_dim=768)

    responder = (
        GuardedLLMResponseGenerator(
            model_name=response_model,
            api_token=api_token,
            allow_fallback=allow_fallback,
            use_llm_for_high_risk=False,
        )
        if response_backend == "guarded_llm"
        else TemplateResponseGenerator()
    )

    return DepressionRiskPipeline(
        encoder=encoder,
        fusion=fusion,
        policy=SafetyPolicyEngine(low=0.30, high=0.70),
        responder=responder,
    )
