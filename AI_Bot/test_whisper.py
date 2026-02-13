from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]

# Choose model size: tiny → fastest, small → good balance, medium/large-v3 → best quality
model_size = "small"

# device="auto" will use GPU if available, otherwise CPU
model = WhisperModel(model_size, device="cpu", compute_type="default")

# Replace with path to any short audio file you have (wav, mp3, m4a, etc.)
audio_file = "Recording_test.m4a"          # ← change this !!

segments, info = model.transcribe(
    audio_file,
    beam_size=5,
    language="en",                 # change to "hi", "bn" etc. if needed
    vad_filter=True                # helps remove silence
)

print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
print(f"Duration: {info.duration:.2f} seconds\n")

for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")