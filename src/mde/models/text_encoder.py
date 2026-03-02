from __future__ import annotations

import math
import re
from typing import Optional

from mde.core.types import ModalityFeatures, UserInput

try:
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    AutoModel = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


class TextEncoder:
    """Hugging Face text encoder wrapper with optional risk head."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        risk_model_name: Optional[str] = "j-hartmann/emotion-english-distilroberta-base",
        max_length: int = 256,
        device: Optional[str] = None,
        local_files_only: bool = False,
        allow_fallback: bool = True,
        load_pretrained: bool = True,
    ) -> None:
        self.max_length = max_length
        self.allow_fallback = allow_fallback
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")

        self.embedding_model_name = embedding_model_name
        self.risk_model_name = risk_model_name

        self.embedding_tokenizer = None
        self.embedding_model = None

        self.risk_tokenizer = None
        self.risk_model = None

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

        self._hf_ready = load_pretrained and self._load_models(local_files_only=local_files_only)

    def _load_models(self, local_files_only: bool) -> bool:
        if torch is None or AutoTokenizer is None or AutoModel is None:
            return False

        try:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name,
                local_files_only=local_files_only,
            )
            self.embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name,
                local_files_only=local_files_only,
            ).to(self.device)
            self.embedding_model.eval()

            if self.risk_model_name:
                self.risk_tokenizer = AutoTokenizer.from_pretrained(
                    self.risk_model_name,
                    local_files_only=local_files_only,
                )
                self.risk_model = AutoModelForSequenceClassification.from_pretrained(
                    self.risk_model_name,
                    local_files_only=local_files_only,
                ).to(self.device)
                self.risk_model.eval()
            return True
        except Exception:
            if not self.allow_fallback:
                raise
            return False

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

    def _mean_pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _label_weighted_risk(self, probs: list[float], id2label: dict[int, str]) -> float:
        if not probs or not id2label:
            return 0.0

        weighted = 0.0
        total = 0.0
        for i, p in enumerate(probs):
            label = id2label.get(i, "").lower()
            w = 0.0
            if any(k in label for k in ["sad", "grief", "disappoint", "fear", "guilt", "remorse"]):
                w = 1.0
            elif any(k in label for k in ["neutral", "joy", "optim", "love"]):
                w = 0.0
            if "depress" in label or "suic" in label:
                w = 1.4
            weighted += w * p
            total += p

        if total <= 0.0:
            return 0.0
        return max(0.0, min(1.0, weighted / total))

    def _encode_pretrained(self, text: str) -> tuple[list[float], float]:
        assert torch is not None
        assert self.embedding_model is not None and self.embedding_tokenizer is not None

        emb_inputs = self.embedding_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        emb_inputs = {k: v.to(self.device) for k, v in emb_inputs.items()}

        with torch.no_grad():
            emb_out = self.embedding_model(**emb_inputs)
            pooled = self._mean_pool(emb_out.last_hidden_state, emb_inputs["attention_mask"])
            embedding = pooled.squeeze(0).detach().cpu().tolist()

        risk = 0.0
        if self.risk_model is not None and self.risk_tokenizer is not None:
            risk_inputs = self.risk_tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            risk_inputs = {k: v.to(self.device) for k, v in risk_inputs.items()}

            with torch.no_grad():
                risk_out = self.risk_model(**risk_inputs)
                probs = torch.softmax(risk_out.logits.squeeze(0), dim=-1).detach().cpu().tolist()
            id2label = getattr(self.risk_model.config, "id2label", {})
            risk = self._label_weighted_risk(probs, id2label)

        return embedding, risk

    def encode(self, user_input: UserInput) -> ModalityFeatures:
        text = user_input.text.strip()
        if self._hf_ready:
            text_embedding, text_risk = self._encode_pretrained(text)
            return ModalityFeatures(text_embedding=text_embedding, text_risk=text_risk)

        tokens = self._normalize(text)
        return ModalityFeatures(
            text_embedding=self._hash_embedding(tokens),
            text_risk=self._lexicon_risk(tokens),
        )
