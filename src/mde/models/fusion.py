from __future__ import annotations

from mde.core.types import FusionOutput, ModalityFeatures
from mde.utils.math_utils import sigmoid


class MaskedFusionMLP:
    """Simple fusion head with modality mask features."""

    def __init__(self, text_dim: int = 16, audio_dim: int = 8, visual_dim: int = 8) -> None:
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim

        # Fixed coefficients as a deterministic placeholder for a trained head.
        self.bias = -0.9
        self.w_text = 1.3
        self.w_audio = 0.6
        self.w_visual = 0.5
        self.w_missing = -0.12

    def _pad(self, emb: list[float] | None, size: int) -> tuple[list[float], int]:
        if emb is None:
            return [0.0] * size, 0
        emb = emb[:size]
        if len(emb) < size:
            emb = emb + [0.0] * (size - len(emb))
        return emb, 1

    def predict(self, features: ModalityFeatures) -> FusionOutput:
        t_emb, t_mask = self._pad(features.text_embedding, self.text_dim)
        a_emb, a_mask = self._pad(features.audio_embedding, self.audio_dim)
        v_emb, v_mask = self._pad(features.visual_embedding, self.visual_dim)

        text_signal = sum(abs(v) for v in t_emb) / self.text_dim
        audio_signal = sum(abs(v) for v in a_emb) / self.audio_dim
        visual_signal = sum(abs(v) for v in v_emb) / self.visual_dim
        text_risk = features.text_risk or 0.0

        missing = 2 - (a_mask + v_mask)
        logit = (
            self.bias
            + self.w_text * (0.6 * text_signal + 0.4 * text_risk)
            + self.w_audio * audio_signal
            + self.w_visual * visual_signal
            + self.w_missing * missing
        )
        risk_score = sigmoid(logit)

        return FusionOutput(
            joint_embedding=t_emb + a_emb + v_emb,
            modality_mask=[t_mask, a_mask, v_mask],
            risk_score=risk_score,
        )
