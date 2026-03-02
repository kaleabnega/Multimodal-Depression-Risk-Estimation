from __future__ import annotations

import re

from mde.core.types import ModalityFeatures, TurnInput
from mde.utils.math_utils import sigmoid


class TextEncoder:
    """Lightweight stand-in for a pretrained text encoder."""

    def __init__(self, emb_dim: int = 16) -> None:
        self.emb_dim = emb_dim
        self.risk_lexicon = {
            "hopeless": 1.2,
            "worthless": 1.4,
            "tired": 0.5,
            "empty": 0.8,
            "sad": 0.8,
            "alone": 0.6,
            "suicide": 2.2,
            "kill": 2.0,
            "die": 1.8,
        }

    def _normalize(self, text: str) -> list[str]:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [tok for tok in text.split() if tok]

    def _embed(self, tokens: list[str]) -> list[float]:
        if not tokens:
            return [0.0] * self.emb_dim
        vec = [0.0] * self.emb_dim
        for token in tokens:
            idx = hash(token) % self.emb_dim
            vec[idx] += 1.0
        scale = max(len(tokens), 1)
        return [v / scale for v in vec]

    def _risk(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        score = sum(self.risk_lexicon.get(tok, 0.0) for tok in tokens)
        # Short-text normalization keeps risk stable across message length.
        normalized = score / (len(tokens) ** 0.5)
        return sigmoid(normalized - 0.7)

    def encode(self, turn: TurnInput) -> ModalityFeatures:
        tokens = self._normalize(turn.text)
        return ModalityFeatures(
            text_embedding=self._embed(tokens),
            text_risk=self._risk(tokens),
        )
