from __future__ import annotations

import math
import re
from typing import Any, Optional

from mde.core.types import ModalityFeatures, UserInput

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional runtime dependency
    InferenceClient = None


class HFAPITextEncoder:
    """Hugging Face Inference API wrapper for text embedding + risk prior."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        risk_model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        api_token: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.risk_model_name = risk_model_name
        self.allow_fallback = allow_fallback

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

        self.client = InferenceClient(token=api_token) if InferenceClient is not None else None

    def _normalize(self, text: str) -> list[str]:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [tok for tok in text.split() if tok]

    def _lexicon_risk(self, tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        score = sum(self.risk_lexicon.get(tok, 0.0) for tok in tokens)
        normalized = score / (len(tokens) ** 0.5)
        return 1.0 / (1.0 + math.exp(-(normalized - 0.7)))

    def _hash_embedding(self, tokens: list[str], emb_dim: int = 384) -> list[float]:
        if not tokens:
            return [0.0] * emb_dim
        vec = [0.0] * emb_dim
        for token in tokens:
            idx = hash(token) % emb_dim
            vec[idx] += 1.0
        scale = max(len(tokens), 1)
        return [v / scale for v in vec]

    def _to_list(self, value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def _mean_pool_tokens(self, token_embeddings: Any) -> list[float]:
        data = self._to_list(token_embeddings)
        if not data:
            return [0.0] * 384

        if isinstance(data[0], (int, float)):
            return [float(x) for x in data]

        num_tokens = len(data)
        dim = len(data[0]) if num_tokens > 0 else 384
        out = [0.0] * dim
        for token_vec in data:
            for i, val in enumerate(token_vec):
                out[i] += float(val)

        denom = float(max(num_tokens, 1))
        return [x / denom for x in out]

    def _risk_from_labels(self, labels: Any) -> float:
        rows = self._to_list(labels)
        if not rows:
            return 0.0

        if isinstance(rows, list) and rows and isinstance(rows[0], list):
            rows = rows[0]

        weighted = 0.0
        total = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label", "")).lower()
            score = float(row.get("score", 0.0))

            weight = 0.0
            if any(k in label for k in ["sad", "grief", "disappoint", "fear", "guilt", "remorse"]):
                weight = 1.0
            elif any(k in label for k in ["neutral", "joy", "optim", "love"]):
                weight = 0.0
            if "depress" in label or "suic" in label:
                weight = 1.4

            weighted += weight * score
            total += score

        if total <= 0.0:
            return 0.0
        return max(0.0, min(1.0, weighted / total))

    def encode(self, user_input: UserInput) -> ModalityFeatures:
        text = user_input.text.strip()
        tokens = self._normalize(text)

        if self.client is None:
            return ModalityFeatures(
                text_embedding=self._hash_embedding(tokens),
                text_risk=self._lexicon_risk(tokens),
            )

        try:
            emb_raw = self.client.feature_extraction(
                text,
                model=self.embedding_model_name,
            )
            embedding = self._mean_pool_tokens(emb_raw)

            risk_rows = self.client.text_classification(
                text,
                model=self.risk_model_name,
                top_k=None,
            )
            risk = self._risk_from_labels(risk_rows)
            return ModalityFeatures(text_embedding=embedding, text_risk=risk)
        except Exception:
            if not self.allow_fallback:
                raise
            return ModalityFeatures(
                text_embedding=self._hash_embedding(tokens),
                text_risk=self._lexicon_risk(tokens),
            )
