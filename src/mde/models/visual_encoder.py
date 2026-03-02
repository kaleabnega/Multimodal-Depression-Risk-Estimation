from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    Image = None
    AutoImageProcessor = None
    AutoModel = None
    AutoModelForImageClassification = None


class VisualEncoder:
    """Hugging Face visual encoder wrapper for frame embeddings + affect."""

    def __init__(
        self,
        embedding_model_name: str = "google/vit-base-patch16-224-in21k",
        affect_model_name: str = "trpakov/vit-face-expression",
        device: Optional[str] = None,
        local_files_only: bool = False,
        allow_fallback: bool = True,
        load_pretrained: bool = True,
    ) -> None:
        self.allow_fallback = allow_fallback
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")

        self.embedding_processor = None
        self.embedding_model = None
        self.affect_processor = None
        self.affect_model = None

        self._hf_ready = load_pretrained and self._load_models(
            embedding_model_name=embedding_model_name,
            affect_model_name=affect_model_name,
            local_files_only=local_files_only,
        )

    def _load_models(self, embedding_model_name: str, affect_model_name: str, local_files_only: bool) -> bool:
        if torch is None or AutoImageProcessor is None or AutoModel is None:
            return False

        try:
            self.embedding_processor = AutoImageProcessor.from_pretrained(
                embedding_model_name,
                local_files_only=local_files_only,
            )
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model_name,
                local_files_only=local_files_only,
            ).to(self.device)
            self.embedding_model.eval()

            self.affect_processor = AutoImageProcessor.from_pretrained(
                affect_model_name,
                local_files_only=local_files_only,
            )
            self.affect_model = AutoModelForImageClassification.from_pretrained(
                affect_model_name,
                local_files_only=local_files_only,
            ).to(self.device)
            self.affect_model.eval()
            return True
        except Exception:
            if not self.allow_fallback:
                raise
            return False

    def _load_pil_frames(self, frames: list[str]) -> list[Image.Image]:
        if Image is None:
            return []

        images: list[Image.Image] = []
        for frame in frames:
            path = Path(frame)
            if path.exists() and path.is_file():
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
        return images

    def _keyword_fallback(self, frames: list[str]) -> tuple[list[float], list[float], str]:
        lowered = [f.lower() for f in frames]
        downcast = sum(1 for f in lowered if "down" in f or "avoid" in f)
        smile = sum(1 for f in lowered if "smile" in f)
        frozen = sum(1 for f in lowered if "still" in f or "flat" in f)

        total = max(len(frames), 1)
        down_ratio = downcast / total
        smile_ratio = smile / total
        frozen_ratio = frozen / total

        emb = [0.0] * 8
        emb[0] = down_ratio
        emb[1] = smile_ratio
        emb[2] = frozen_ratio

        sad = min(1.0, max(0.0, 0.6 * down_ratio + 0.5 * frozen_ratio - 0.4 * smile_ratio))
        engaged = min(1.0, max(0.0, 0.6 * smile_ratio))
        neutral = min(1.0, max(0.0, 1.0 - sad - engaged))

        affect = [sad, neutral, engaged]
        summary = "reduced expressivity and downward gaze" if sad > 0.45 else "neutral facial affect"
        return emb, affect, summary

    def _map_affect_probs(self, probs: list[float], id2label: dict[int, str]) -> list[float]:
        sad = 0.0
        neutral = 0.0
        engaged = 0.0

        for idx, prob in enumerate(probs):
            label = id2label.get(idx, "").lower()
            if "sad" in label or "fear" in label or "disgust" in label:
                sad += prob
            elif "neutral" in label:
                neutral += prob
            elif any(k in label for k in ["happy", "surprise"]):
                engaged += prob

        unknown = max(0.0, 1.0 - (sad + neutral + engaged))
        neutral += unknown

        total = sad + neutral + engaged
        if total <= 0.0:
            return [0.0, 1.0, 0.0]
        return [sad / total, neutral / total, engaged / total]

    def _summary(self, affect_probs: list[float]) -> str:
        sad, neutral, engaged = affect_probs
        if sad >= max(neutral, engaged):
            return "reduced expressivity and downward gaze"
        if engaged > neutral:
            return "higher facial expressivity"
        return "neutral facial affect"

    def encode(self, frames: list[str]) -> tuple[list[float], list[float], str]:
        if not frames:
            return [0.0] * 8, [0.0, 1.0, 0.0], "no visual cues"

        pil_frames = self._load_pil_frames(frames)
        if not self._hf_ready or not pil_frames:
            return self._keyword_fallback(frames)

        assert torch is not None
        assert self.embedding_model is not None and self.embedding_processor is not None
        assert self.affect_model is not None and self.affect_processor is not None

        emb_inputs = self.embedding_processor(images=pil_frames, return_tensors="pt")
        emb_inputs = {k: v.to(self.device) for k, v in emb_inputs.items()}

        with torch.no_grad():
            emb_out = self.embedding_model(**emb_inputs)
            frame_emb = emb_out.last_hidden_state[:, 0, :]
            emb = frame_emb.mean(dim=0).detach().cpu().tolist()

        affect_inputs = self.affect_processor(images=pil_frames, return_tensors="pt")
        affect_inputs = {k: v.to(self.device) for k, v in affect_inputs.items()}

        with torch.no_grad():
            affect_out = self.affect_model(**affect_inputs)
            probs = torch.softmax(affect_out.logits, dim=-1).mean(dim=0).detach().cpu().tolist()

        id2label = getattr(self.affect_model.config, "id2label", {})
        affect_probs = self._map_affect_probs(probs, id2label)
        return emb, affect_probs, self._summary(affect_probs)
