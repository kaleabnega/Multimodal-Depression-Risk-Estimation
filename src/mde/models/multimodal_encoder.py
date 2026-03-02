from __future__ import annotations

from mde.core.types import ModalityFeatures, TurnInput
from mde.models.audio_encoder import AudioEncoder
from mde.models.text_encoder import TextEncoder
from mde.models.visual_encoder import VisualEncoder


class MultimodalEncoder:
    """Runs each modality branch and returns unified feature object."""

    def __init__(
        self,
        text_encoder: TextEncoder,
        audio_encoder: AudioEncoder,
        visual_encoder: VisualEncoder,
    ) -> None:
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder

    def encode(self, turn: TurnInput) -> tuple[ModalityFeatures, str | None, str | None]:
        features = self.text_encoder.encode(turn)
        audio_summary = None
        visual_summary = None

        if turn.audio is not None:
            a_emb, a_affect, audio_summary = self.audio_encoder.encode(turn.audio)
            features.audio_embedding = a_emb
            features.audio_affect_probs = a_affect

        if turn.frames is not None:
            v_emb, v_affect, visual_summary = self.visual_encoder.encode(turn.frames)
            features.visual_embedding = v_emb
            features.visual_affect_probs = v_affect

        return features, audio_summary, visual_summary
