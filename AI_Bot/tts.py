"""Kokoro-82M TTS: text → audio array or WAV file. Used by voice bot pipeline."""
from __future__ import annotations

import os
import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np

# Suppress noisy Kokoro/PyTorch warnings (dropout, weight_norm deprecation)
warnings.filterwarnings("ignore", message=".*dropout option adds dropout after all but last recurrent layer.*", module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", message=".*weight_norm.*is deprecated.*", category=FutureWarning, module="torch.nn.utils.weight_norm")

if TYPE_CHECKING:
    from kokoro import KPipeline

# Kokoro sample rate (fixed)
SAMPLE_RATE = 24_000

# Default: American English, af_heart voice
DEFAULT_LANG = "b"
DEFAULT_VOICE = "bm_george"

# TTS device: "cuda", "cpu", or None. Default "cpu" to avoid cudnnGetLibConfig Error 127 on Windows. Set env TTS_DEVICE=cuda to try GPU.
TTS_DEVICE: str | None = os.environ.get("TTS_DEVICE", None)
if TTS_DEVICE is not None:
    TTS_DEVICE = TTS_DEVICE.strip().lower() or None

# Cached pipeline for (lang, voice) to reduce latency after first call
_pipeline_cache: dict[tuple[str, str], KPipeline] = {}


def _ensure_pytorch_dll_path() -> None:
    """On Windows, add PyTorch's bundled CUDA/cuDNN to DLL search path to fix cudnnGetLibConfig Error 127."""
    if sys.platform != "win32":
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.origin is None:
            return
        torch_dir = os.path.dirname(spec.origin)
        torch_lib = os.path.join(torch_dir, "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
    except Exception:
        pass


# Run once at import so PyTorch's DLLs are found before Kokoro loads CUDA (Windows cuDNN fix)
if TTS_DEVICE != "cpu":
    _ensure_pytorch_dll_path()


def _get_pipeline(lang_code: str, voice: str) -> KPipeline:
    from kokoro import KPipeline
    import torch

    key = (lang_code, voice)
    if key not in _pipeline_cache:
        # Default CPU: cu121 PyTorch triggers cudnnGetLibConfig Error 127 for Kokoro on this Windows setup. Set TTS_DEVICE=cuda to try GPU.
        device = TTS_DEVICE if TTS_DEVICE is not None else "cpu"
        _pipeline_cache[key] = KPipeline(lang_code=lang_code, device=device, repo_id="hexgrad/Kokoro-82M")
    return _pipeline_cache[key]


def text_to_speech(
    text: str,
    *,
    voice: str = DEFAULT_VOICE,
    lang_code: str = DEFAULT_LANG,
    save_path: str | None = None,
) -> tuple[int, np.ndarray]:
    """
    Synthesize speech from text using Kokoro-82M.

    Args:
        text: Input text to speak.
        voice: Kokoro voice id (e.g. 'af_heart' for US English).
        lang_code: Language code ('a' = American English, 'b' = British, etc.).
        save_path: If set, also write WAV to this path.

    Returns:
        (sample_rate, audio) where audio is float32 numpy array in [-1, 1].
    """
    import soundfile as sf

    if not text or not text.strip():
        return SAMPLE_RATE, np.array([], dtype=np.float32)

    pipeline = _get_pipeline(lang_code, voice)
    chunks: list[np.ndarray] = []
    for _gs, _ps, audio in pipeline(text.strip(), voice=voice):
        chunks.append(audio)

    if not chunks:
        return SAMPLE_RATE, np.array([], dtype=np.float32)

    audio = np.concatenate(chunks, axis=0)
    if save_path:
        sf.write(save_path, audio, SAMPLE_RATE)
    return SAMPLE_RATE, audio
