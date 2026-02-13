"""
Standalone test for Kokoro-82M TTS.
Run: python test_tts.py
- Generates a short WAV file and plays it.
- No dependency on llm or live_transcript; only kokoro + soundfile + sounddevice.
"""
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 24_000
OUTPUT_WAV = "test_kokoro_out.wav"

def main():
    print("Loading Kokoro pipeline (first run may download model)...")
    pipeline = KPipeline(lang_code="a")  # American English

    text = "Hello. I am TONY, a voice assistant. This is Kokoro text to speech."
    print(f"Generating speech for: {text!r}")
    chunks = []
    for _i, (_gs, _ps, audio) in enumerate(pipeline(text, voice="bm_george")):
        chunks.append(audio)

    if not chunks:
        print("No audio generated.")
        return
    audio = np.concatenate(chunks, axis=0)

    sf.write(OUTPUT_WAV, audio, SAMPLE_RATE)
    print(f"Saved: {OUTPUT_WAV}")

    print("Playing...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
    print("Done.")

if __name__ == "__main__":
    main()
