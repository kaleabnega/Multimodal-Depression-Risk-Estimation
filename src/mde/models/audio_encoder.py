from __future__ import annotations

from typing import Optional

try:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    AutoFeatureExtractor = None
    AutoModel = None
    AutoModelForAudioClassification = None


class AudioEncoder:
    """Hugging Face audio encoder wrapper for speech embeddings + affect."""

    def __init__(
        self,
        embedding_model_name: str = "facebook/wav2vec2-base",
        affect_model_name: str = "superb/wav2vec2-base-superb-er",
        sample_rate: int = 16_000,
        device: Optional[str] = None,
        local_files_only: bool = False,
        allow_fallback: bool = True,
        load_pretrained: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.allow_fallback = allow_fallback
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")

        self.embedding_model = None
        self.embedding_extractor = None
        self.affect_model = None
        self.affect_extractor = None

        self._hf_ready = load_pretrained and self._load_models(
            embedding_model_name=embedding_model_name,
            affect_model_name=affect_model_name,
            local_files_only=local_files_only,
        )

    def _load_models(self, embedding_model_name: str, affect_model_name: str, local_files_only: bool) -> bool:
        if torch is None or AutoFeatureExtractor is None or AutoModel is None:
            return False

        try:
            self.embedding_extractor = AutoFeatureExtractor.from_pretrained(
                embedding_model_name,
                local_files_only=local_files_only,
            )
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model_name,
                local_files_only=local_files_only,
            ).to(self.device)
            self.embedding_model.eval()

            self.affect_extractor = AutoFeatureExtractor.from_pretrained(
                affect_model_name,
                local_files_only=local_files_only,
            )
            self.affect_model = AutoModelForAudioClassification.from_pretrained(
                affect_model_name,
                local_files_only=local_files_only,
            ).to(self.device)
            self.affect_model.eval()
            return True
        except Exception:
            if not self.allow_fallback:
                raise
            return False

    def _map_affect_probs(self, probs: list[float], id2label: dict[int, str]) -> list[float]:
        sad = 0.0
        neutral = 0.0
        activated = 0.0

        for idx, prob in enumerate(probs):
            label = id2label.get(idx, "").lower()
            if "sad" in label:
                sad += prob
            elif "neutral" in label or "calm" in label:
                neutral += prob
            elif any(k in label for k in ["ang", "hap", "exc", "surp"]):
                activated += prob

        unknown = max(0.0, 1.0 - (sad + neutral + activated))
        neutral += unknown

        total = sad + neutral + activated
        if total <= 0.0:
            return [0.0, 1.0, 0.0]
        return [sad / total, neutral / total, activated / total]

    def _summary(self, affect_probs: list[float]) -> str:
        sad, neutral, activated = affect_probs
        if sad >= max(neutral, activated):
            return "flat tone and slower cadence"
        if activated > neutral:
            return "elevated vocal activation"
        return "neutral vocal affect"

    def _encode_fallback(self, waveform: list[float]) -> tuple[list[float], list[float], str]:
        if not waveform:
            return [0.0] * 8, [0.0, 1.0, 0.0], "no audio cues"

        abs_mean = sum(abs(x) for x in waveform) / len(waveform)
        dynamic_range = max(waveform) - min(waveform)
        pause_proxy = sum(1 for x in waveform if abs(x) < 0.02) / len(waveform)

        emb = [0.0] * 8
        emb[0] = abs_mean
        emb[1] = dynamic_range
        emb[2] = pause_proxy

        sad = min(1.0, max(0.0, 0.7 * pause_proxy + 0.5 * (0.15 - abs_mean)))
        activated = min(1.0, max(0.0, 0.8 * dynamic_range - 0.2))
        neutral = min(1.0, max(0.0, 1.0 - sad - activated))

        affect = [sad, neutral, activated]
        return emb, affect, self._summary(affect)

    def encode(self, waveform: list[float]) -> tuple[list[float], list[float], str]:
        if not waveform:
            return [0.0] * 8, [0.0, 1.0, 0.0], "no audio cues"

        if not self._hf_ready:
            return self._encode_fallback(waveform)

        assert torch is not None
        assert self.embedding_extractor is not None and self.embedding_model is not None
        assert self.affect_extractor is not None and self.affect_model is not None

        emb_inputs = self.embedding_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        emb_inputs = {k: v.to(self.device) for k, v in emb_inputs.items()}

        with torch.no_grad():
            emb_out = self.embedding_model(**emb_inputs)
            emb = emb_out.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().tolist()

        affect_inputs = self.affect_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        affect_inputs = {k: v.to(self.device) for k, v in affect_inputs.items()}

        with torch.no_grad():
            affect_out = self.affect_model(**affect_inputs)
            probs = torch.softmax(affect_out.logits.squeeze(0), dim=-1).detach().cpu().tolist()

        id2label = getattr(self.affect_model.config, "id2label", {})
        affect_probs = self._map_affect_probs(probs, id2label)
        return emb, affect_probs, self._summary(affect_probs)
