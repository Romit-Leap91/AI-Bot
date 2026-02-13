# live_transcript.py — Faster-Whisper → LLM (Qwen/Ollama)
# LLM replies only after user stops speaking (~2s silence).
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from llm import ask_llm, SYSTEM_PROMPT
import queue
import sys
import time

# Config
MODEL_SIZE = "small"          # tiny / base / small / medium
DEVICE = "cuda"               # "cuda" if GPU
BLOCK_SIZE = 8000             # ~0.5s at 16kHz
SAMPLE_RATE = 16000
LANGUAGE = "en"               # "hi", "bn", None=auto

# When user stops: treat 2 seconds of silence as end of utterance, then run STT + LLM
SILENCE_DURATION_SEC = 2.0
SILENCE_THRESHOLD_RMS = 0.01  # RMS below this = silence (tune if needed: 0.005–0.02)
MIN_SPEECH_SEC = 0.5          # Ignore if utterance is shorter than this

BLOCKS_PER_SEC = SAMPLE_RATE / BLOCK_SIZE  # ~2 blocks per second
SILENCE_BLOCKS = int(SILENCE_DURATION_SEC * BLOCKS_PER_SEC)  # ~4 blocks for 2s

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def is_silence(chunk: np.ndarray, threshold: float) -> bool:
    rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
    return rms < threshold

print("Loading model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="float16")
print("Model ready. Speak now — I'll reply after you pause ~2 seconds. (Ctrl+C to stop)")

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
                
                # User stopped: 2s of silence after some speech → transcribe + LLM
                if silence_block_count >= SILENCE_BLOCKS and had_speech:
                    # Utterance = everything except the trailing silence
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
                                reply = ask_llm(transcript, system=SYSTEM_PROMPT)
                                print(f"  → TONY: {reply}\n")
                            except Exception as e:
                                print(f"  → LLM error: {e}\n", file=sys.stderr)
                    
                    # Reset: keep last 0.5s to avoid clipping next utterance
                    buffer = buffer[-BLOCK_SIZE:]
                    silence_block_count = 0
                    had_speech = False
                    
            except queue.Empty:
                time.sleep(0.05)
                continue

except KeyboardInterrupt:
    print("\nStopped.")