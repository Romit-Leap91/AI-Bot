"""Faster-Whisper STT: audio array → text. Used by voice bot pipeline."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

# Config (can override via env)
MODEL_SIZE = os.environ.get("STT_MODEL_SIZE", "small")
DEVICE = os.environ.get("STT_DEVICE", "cuda")
COMPUTE_TYPE = "float16"
SAMPLE_RATE = 16000
DEFAULT_LANGUAGE = "en"

_model: Any = None


def load_model():
    """Load Whisper model once. Called by bot at startup or on first transcribe."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def transcribe(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    language: str | None = DEFAULT_LANGUAGE,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> tuple[str, str]:
    """
    Transcribe audio to text.

    Args:
        audio: Float32 numpy array (mono).
        sample_rate: Audio sample rate (default 16000).
        language: Language code or None for auto.
        beam_size: Decoding beam size.
        vad_filter: Use VAD to skip silence.

    Returns:
        (transcript_text, detected_language)
    """
    model = load_model()
    segments, info = model.transcribe(
        audio,
        beam_size=beam_size,
        language=language,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    lines = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            lines.append(t)
    transcript = " ".join(lines).strip()
    lang = getattr(info, "language", "en") or "en"
    return transcript, lang
