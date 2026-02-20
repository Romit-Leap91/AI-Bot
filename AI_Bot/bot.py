# bot.py — Voice bot: STT → LLM (OpenAI) → TTS. Run this to activate the bot.

import os
import sys
import queue
import time

import numpy as np
import sounddevice as sd

# Fix CUDA/cuDNN on Windows
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

from stt import load_model, transcribe, SAMPLE_RATE
from llm import ask_llm, SYSTEM_PROMPT
from tts import text_to_speech
from db import ping_sync

# --- Pipeline config ---
BLOCK_SIZE = 8000
SILENCE_DURATION_SEC = 3.0
SILENCE_THRESHOLD_RMS = 0.01
MIN_SPEECH_SEC = 0.5
LANGUAGE = "en"
ENABLE_TTS = True

BLOCKS_PER_SEC = SAMPLE_RATE / BLOCK_SIZE
SILENCE_BLOCKS = int(SILENCE_DURATION_SEC * BLOCKS_PER_SEC)

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


def run():
    print("Loading model...")
    load_model()

    if ping_sync():
        print("DB: MongoDB Atlas connected. LLM can query your data (e.g. CrewgleAI_Store, Sports Items).")
    else:
        print("DB: MongoDB not reachable — LLM will answer from its knowledge only.", file=sys.stderr)

    print("Model ready. Speak now — I'll reply (text + speech) after you pause ~3s. (Ctrl+C to stop)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    ):
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
                        transcript, lang = transcribe(utterance, language=LANGUAGE)
                        print(f"[lang: {lang}]")
                        if transcript:
                            print(transcript)
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


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nStopped.")
