# live_transcript.py — STT → LLM (OpenAI) → TTS → output
# LLM queries MongoDB Atlas when user asks about stored data; otherwise answers from its knowledge.

# Fix CUDA/cuDNN on Windows
import os
import sys
if sys.platform == "win32":
    try:
        import importlib.util
        _spec = importlib.util.find_spec("torch")
        if _spec is not None and _spec.origin is not None:
            _torch_lib = os.path.join(os.path.dirname(_spec.origin), "lib")
            if os.path.isdir(_torch_lib):
                os.add_dll_directory(_torch_lib)
    except Exception:
        pass

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from llm import ask_llm, SYSTEM_PROMPT
from tts import text_to_speech
from db import ping_sync
import queue
import time

# --- STT config ---
MODEL_SIZE = "small"
DEVICE = "cuda"
BLOCK_SIZE = 8000
SAMPLE_RATE = 16000
LANGUAGE = "en"

SILENCE_DURATION_SEC = 3.0
SILENCE_THRESHOLD_RMS = 0.01
MIN_SPEECH_SEC = 0.5

BLOCKS_PER_SEC = SAMPLE_RATE / BLOCK_SIZE
SILENCE_BLOCKS = int(SILENCE_DURATION_SEC * BLOCKS_PER_SEC)

ENABLE_TTS = True

audio_queue = queue.Queue()


def play_tts_reply(reply: str) -> None:
    if not ENABLE_TTS or not (reply and reply.strip()):
        return
    try:
        sr, audio = text_to_speech(reply)
        if audio.size > 0:
            sd.play(audio, sr)
            sd.wait()
    except Exception as e:
        print(f"  → TTS error: {e}\n", file=sys.stderr)


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())


def is_silence(chunk: np.ndarray, threshold: float) -> bool:
    rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
    return rms < threshold


print("Loading model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="float16")

# MongoDB Atlas: LLM can query when user asks about stored data
if ping_sync():
    print("DB: MongoDB Atlas connected. LLM can query your data (e.g. CrewgleAI_Store, Sports Items).")
else:
    print("DB: MongoDB not reachable — LLM will answer from its knowledge only.", file=sys.stderr)

print("Model ready. Speak now — I'll reply (text + speech) after you pause ~3s. (Ctrl+C to stop)")

try:
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype='float32',
                        blocksize=BLOCK_SIZE,
                        callback=audio_callback):

        buffer = np.array([], dtype=np.float32)
        silence_block_count = 0
        had_speech = False

        while True:
            try:
                chunk = audio_queue.get(timeout=0.1)
                chunk_flat = chunk.flatten()
                buffer = np.concatenate((buffer, chunk_flat))

                if is_silence(chunk_flat, SILENCE_THRESHOLD_RMS):
                    silence_block_count += 1
                else:
                    silence_block_count = 0
                    had_speech = True

                if silence_block_count >= SILENCE_BLOCKS and had_speech:
                    silence_samples = int(SAMPLE_RATE * SILENCE_DURATION_SEC)
                    utterance = buffer[:-silence_samples] if len(buffer) > silence_samples else buffer

                    if len(utterance) >= SAMPLE_RATE * MIN_SPEECH_SEC:
                        segments, info = model.transcribe(
                            utterance,
                            beam_size=5,
                            language=LANGUAGE,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500),
                        )
                        print(f"[lang: {info.language}]")
                        transcript_lines = []
                        for segment in segments:
                            text = segment.text.strip()
                            if text:
                                print(f"[{segment.start:.1f}s → {segment.end:.1f}s] {text}")
                                transcript_lines.append(text)
                        transcript = " ".join(transcript_lines).strip()
                        if transcript:
                            try:
                                reply = ask_llm(transcript, system=SYSTEM_PROMPT, use_tools=True)
                                print(f"  → TONY: {reply}\n")
                                play_tts_reply(reply)
                            except Exception as e:
                                print(f"  → LLM error: {e}\n", file=sys.stderr)

                    buffer = buffer[-BLOCK_SIZE:]
                    silence_block_count = 0
                    had_speech = False

            except queue.Empty:
                time.sleep(0.05)
                continue

except KeyboardInterrupt:
    print("\nStopped.")
