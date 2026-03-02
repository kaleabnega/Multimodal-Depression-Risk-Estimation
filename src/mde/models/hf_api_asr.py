from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional runtime dependency
    InferenceClient = None


class HFAPIAudioTranscriber:
    """Hugging Face Inference API ASR wrapper."""

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        api_token: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self.client = InferenceClient(token=api_token) if InferenceClient is not None else None

    def _normalize_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            text = value.get("text") or value.get("generated_text") or ""
            return str(text).strip()
        return str(value).strip()

    def transcribe(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Audio file not found for ASR: {path}")

        if self.client is None:
            if self.allow_fallback:
                return ""
            raise RuntimeError("huggingface_hub is not installed")

        with path.open("rb") as f:
            raw = f.read()

        last_error: Exception | None = None

        try:
            if hasattr(self.client, "automatic_speech_recognition"):
                out = self.client.automatic_speech_recognition(raw, model=self.model_name)
                text = self._normalize_text(out)
                if text:
                    return text
        except Exception as exc:  # pragma: no cover - runtime/provider dependent
            last_error = exc

        # Client compatibility fallback for variants exposing speech_to_text.
        try:
            if hasattr(self.client, "speech_to_text"):
                out = self.client.speech_to_text(raw, model=self.model_name)
                text = self._normalize_text(out)
                if text:
                    return text
        except Exception as exc:  # pragma: no cover - runtime/provider dependent
            last_error = exc

        if self.allow_fallback:
            return ""
        if last_error is not None:
            raise RuntimeError("ASR transcription failed") from last_error
        raise RuntimeError("No compatible ASR method found on InferenceClient")
