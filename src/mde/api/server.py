from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mde.core.types import UserInput
from mde.models.hf_api_asr import HFAPIAudioTranscriber
from mde.services.pipeline import DepressionRiskPipeline
from mde.utils.vision_pipeline import FacePipeline
from mde.runtime import build_pipeline


def _load_dotenv(dotenv_path: str | Path = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (key not in os.environ or not str(os.environ.get(key, "")).strip()):
            os.environ[key] = value


def _read_wav_16khz_mono(path: str) -> list[float]:
    wav_path = Path(path)
    if not wav_path.exists() or not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sample_rate != 16_000:
        raise ValueError(f"Expected 16kHz WAV input, got {sample_rate}Hz at {wav_path}.")

    if sample_width not in (1, 2, 4):
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    samples: list[float] = []
    bytes_per_frame = sample_width * num_channels
    max_int = float((1 << (8 * sample_width - 1)) - 1)

    for i in range(0, len(raw), bytes_per_frame):
        frame_bytes = raw[i : i + bytes_per_frame]
        channel_values: list[float] = []
        for ch in range(num_channels):
            start = ch * sample_width
            chunk = frame_bytes[start : start + sample_width]

            if sample_width == 1:
                value = (chunk[0] - 128) / 128.0
            else:
                intval = int.from_bytes(chunk, byteorder="little", signed=True)
                value = intval / max_int if max_int else 0.0

            channel_values.append(max(-1.0, min(1.0, value)))

        samples.append(sum(channel_values) / max(len(channel_values), 1))

    return samples


class ChatRequest(BaseModel):
    text: str = Field(default="")
    audio_wav: Optional[str] = None
    video: Optional[str] = None
    frames: Optional[list[str]] = None
    asr_from_audio: bool = False
    asr_model: str = "openai/whisper-large-v3"
    video_fps: float = 1.0
    max_frames: int = 10
    skip_face_pipeline: bool = False
    debug: bool = False


class ChatResponse(BaseModel):
    response: str
    risk_score: float
    policy_state: str
    reasons: list[str]
    modality_mask: list[int]
    response_source: str = "unknown"
    response_model: Optional[str] = None
    debug: Optional[dict] = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_load_dotenv(PROJECT_ROOT / ".env")
API_TOKEN = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
RESPONSE_MODEL = os.getenv("MDE_RESPONSE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
PIPELINE: DepressionRiskPipeline = build_pipeline(
    backend="hf_api",
    response_backend="guarded_llm",
    allow_fallback=True,
    api_token=API_TOKEN,
    response_model=RESPONSE_MODEL,
)

app = FastAPI(title="MDE API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "hf_token_loaded": "true" if bool(API_TOKEN) else "false",
        "response_model": RESPONSE_MODEL,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        final_text = req.text.strip()
        audio = _read_wav_16khz_mono(req.audio_wav) if req.audio_wav else None

        if req.asr_from_audio:
            if not req.audio_wav:
                raise ValueError("asr_from_audio=true requires audio_wav path")
            transcriber = HFAPIAudioTranscriber(
                model_name=req.asr_model,
                api_token=API_TOKEN,
                allow_fallback=True,
            )
            transcript = transcriber.transcribe(req.audio_wav).strip()
            if transcript:
                final_text = f"{final_text}\n{transcript}".strip() if final_text else transcript

        if not final_text:
            raise ValueError("text is required (or provide audio with ASR)")

        frames: list[str] | None = None
        face_pipeline = FacePipeline(fps=req.video_fps, max_frames=req.max_frames)

        if req.video:
            frames = face_pipeline.extract_frames_from_video(req.video)

        if req.frames:
            validated = []
            for frame in req.frames:
                frame_path = Path(frame)
                if not frame_path.exists() or not frame_path.is_file():
                    raise FileNotFoundError(f"Frame file not found: {frame_path}")
                validated.append(str(frame_path))
            frames = (frames or []) + validated

        if frames and not req.skip_face_pipeline:
            frames = face_pipeline.process_frames(frames)
        elif frames:
            frames = frames[: req.max_frames]

        user_input = UserInput(text=final_text, audio=audio, frames=frames)
        output = PIPELINE.run_user_input(user_input)
        responder_meta = getattr(PIPELINE.responder, "last_generation_meta", {}) or {}
        response_source = str(responder_meta.get("source", "unknown"))
        response_model = responder_meta.get("model")

        debug = None
        if req.debug:
            debug = {
                "text_len": len(final_text),
                "audio_samples": len(audio) if audio is not None else 0,
                "num_frames_inferred": len(frames) if frames is not None else 0,
                "text_risk": output.features.text_risk,
                "audio_affect_probs": output.features.audio_affect_probs,
                "visual_affect_probs": output.features.visual_affect_probs,
                "responder_meta": responder_meta,
                "hf_token_loaded": bool(API_TOKEN),
                "response_model": RESPONSE_MODEL,
            }

        return ChatResponse(
            response=output.response,
            risk_score=output.fusion.risk_score,
            policy_state=output.policy.state.value,
            reasons=output.policy.reasons,
            modality_mask=output.fusion.modality_mask,
            response_source=response_source,
            response_model=str(response_model) if response_model is not None else None,
            debug=debug,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
