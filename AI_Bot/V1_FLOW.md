# V1 Voice Bot — Flow and Components (1.1, 1.2, 1.3)

This document describes what has been built in V1, in chronological order: how each part works, how they are wired, and where CPU vs GPU is used.

---

## 1. Overview: What V1 Does

V1 is a **local voice bot** that:

1. Listens to your microphone.
2. When you stop speaking for about 2 seconds, it transcribes what you said (Speech-to-Text).
3. Sends that text to a local LLM (Ollama) and gets a reply.
4. Speaks the reply aloud using TTS (Kokoro).

So the flow is: **Microphone → 1.1 STT → 1.2 LLM → 1.3 TTS → Speaker**.

---

## 2. Part 1.1 — STT (Speech-to-Text): Faster Whisper

### What was developed

- **Technology:** Faster Whisper (CTranslate2-based Whisper).
- **Role:** Converts the user’s **voice** (audio) into **text**.
- **Where it runs:** In `live_transcript.py`. The main script captures audio from the mic, detects when you stop speaking, then sends that audio to Faster Whisper for transcription.

### How it works (chronological)

1. **Mic capture**  
   - `sounddevice` opens an audio input stream at 16 kHz, mono, float32.  
   - Every small block of audio is put into a queue.

2. **Silence detection**  
   - Audio is buffered.  
   - When the signal is below a threshold (RMS), it’s treated as silence.  
   - After **about 2 seconds of silence** following some speech, the buffer is treated as one “utterance.”

3. **Transcription**  
   - That utterance (numpy array) is passed to **Faster Whisper** (`WhisperModel`).  
   - Model size is set to `small` (config: `MODEL_SIZE`).  
   - Output: **text** (the transcript) and optional language info.

4. **Output**  
   - The transcript is printed and passed to the next step (LLM).

### Main config (in `live_transcript.py`)

| Config        | Value   | Meaning                          |
|---------------|---------|----------------------------------|
| `MODEL_SIZE`  | `small` | Whisper model size               |
| `DEVICE`      | `cuda`  | Run STT on GPU                   |
| `SAMPLE_RATE` | 16000   | Mic sample rate (Hz)             |
| `LANGUAGE`    | `en`    | Prefer English                   |
| `SILENCE_DURATION_SEC` | 2.0 | Seconds of silence = end of turn |

### Code / files

- **Entry:** `live_transcript.py` — `WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="float16")`, then `model.transcribe(utterance, ...)`.
- **Dependencies:** `faster_whisper`, `sounddevice`, `numpy`.

### Device (CPU vs GPU)

- **Configured:** `DEVICE = "cuda"` → STT runs on **GPU**.
- **Why:** Faster Whisper uses CTranslate2; with CUDA it runs faster. PyTorch is **cu121** (CUDA 12.1); `cublas64_12.dll` and related DLLs come from that build.

---

## 3. Part 1.2 — LLM (Large Language Model): Ollama

### What was developed

- **Technology:** Ollama (local LLM server) + Python `ollama` client.
- **Role:** Takes the **user text** (transcript) and returns **assistant text** (TONY’s reply).
- **Where it runs:** The LLM runs in the **Ollama process** (separate from Python). Our code only sends prompts and reads replies.

### How it works (chronological)

1. **Input**  
   - After 1.1, we have a string: the **transcript** of what the user said.

2. **Call**  
   - `llm.ask_llm(transcript, system=SYSTEM_PROMPT)` is called.  
   - It builds messages: optional system message (“You are TONY, …”) + user message (transcript).  
   - It calls `ollama.chat(model=OLLAMA_MODEL, messages=messages)`.

3. **Ollama**  
   - The Ollama server loads the model (e.g. `ministral-3:3b`) and generates a reply.  
   - Reply is returned as `response["message"]["content"]`.

4. **Output**  
   - The reply **text** is returned to `live_transcript.py`, printed, and then passed to TTS.

### Main config (in `llm.py`)

| Config         | Value              | Meaning                    |
|----------------|--------------------|----------------------------|
| `OLLAMA_MODEL` | `ministral-3:3b`   | Model name in Ollama       |
| `SYSTEM_PROMPT`| “You are TONY…”    | Defines the assistant      |

### Code / files

- **Module:** `llm.py` — `ask_llm(prompt, system=...)` using `ollama.chat(...)`.
- **Used by:** `live_transcript.py` (and optionally `test_llm.py` for CLI testing).
- **Dependencies:** `ollama` (Python package). Ollama server must be installed and running.

### Device (CPU vs GPU)

- **Where it runs:** Inside the **Ollama** process, not in our Python script.
- **Typical setup:** If you have an NVIDIA GPU and didn’t force CPU, Ollama usually uses **GPU** for the model. Our code does not set device; it’s controlled by Ollama.

---

## 4. Part 1.3 — TTS (Text-to-Speech): Kokoro-82M

### What was developed

- **Technology:** Kokoro-82M (PyTorch-based TTS).
- **Role:** Converts the **reply text** from the LLM into **speech** (audio), then we play it.
- **Where it runs:** In our process. `tts.py` exposes `text_to_speech(text)`; `live_transcript.py` calls it and plays the returned audio with `sounddevice`.

### How it works (chronological)

1. **Input**  
   - After 1.2, we have a string: the **LLM reply** (e.g. “Hey bro! What’s up?…”).

2. **Play helper**  
   - `live_transcript.py` calls `play_tts_reply(reply)`.  
   - If TTS is disabled or reply is empty, it returns.  
   - Otherwise it calls `text_to_speech(reply)` from `tts.py`.

3. **Synthesis**  
   - `text_to_speech` uses a **cached** `KPipeline` (one per language/voice).  
   - Pipeline runs Kokoro on the text and returns (sample_rate, audio_array).  
   - Audio is 24 kHz float32.

4. **Playback**  
   - `play_tts_reply` uses `sounddevice`: `sd.play(audio, sr)` then `sd.wait()` so playback finishes before the next turn.

5. **Output**  
   - User **hears** TONY’s reply. The same reply is also **printed** in the terminal.

### Main config (in `tts.py` and `live_transcript.py`)

| Config        | Where       | Value / meaning                          |
|---------------|------------|-------------------------------------------|
| `ENABLE_TTS`  | live_transcript | `True` = speak replies; `False` = text only |
| `TTS_DEVICE`  | env / tts  | `cpu` (default) or `cuda`; env overrides  |
| `DEFAULT_LANG`| tts        | `"b"` (e.g. British English)              |
| `DEFAULT_VOICE` | tts      | `"bm_george"`                             |
| `repo_id`     | tts        | `"hexgrad/Kokoro-82M"` (passed to Kokoro) |

### Code / files

- **Module:** `tts.py` — `text_to_speech(text, voice=..., lang_code=..., save_path=...)`, returns `(sample_rate, audio)`. Uses `KPipeline(lang_code, device, repo_id="hexgrad/Kokoro-82M")`.
- **Used by:** `live_transcript.py` → `play_tts_reply(reply)` → `text_to_speech(reply)` then `sd.play` / `sd.wait`.
- **Dependencies:** `kokoro`, `soundfile`, PyTorch (e.g. cu121). Optional: `test_tts.py` for standalone TTS test.

### Device (CPU vs GPU)

- **Configured default:** **CPU** (`device = TTS_DEVICE if TTS_DEVICE is not None else "cpu"`).
- **Reason:** On this Windows setup, using PyTorch CUDA for Kokoro triggers “Could not load symbol cudnnGetLibConfig. Error code 127”. Running TTS on CPU avoids that.
- **Override:** Set env `TTS_DEVICE=cuda` to try GPU (may hit the same cuDNN error until the environment is fixed).

---

## 5. How the three parts are wired (chronological flow)

High-level flow:

```
[Microphone] → 1.1 STT → [Transcript text] → 1.2 LLM → [Reply text] → 1.3 TTS → [Audio] → [Speaker]
```

Step-by-step in code:

1. **Start**  
   - You run `python live_transcript.py`.  
   - STT model loads (Faster Whisper).  
   - Mic stream starts (sounddevice).

2. **You speak**  
   - Audio is buffered.  
   - When ~2 s of silence is detected after speech, the buffer is sent to Faster Whisper.

3. **1.1 STT**  
   - `model.transcribe(utterance, ...)` → segments.  
   - Transcript = joined segment texts.  
   - Printed (e.g. `[0.0s → 2.0s] Hi bro`).

4. **1.2 LLM**  
   - If transcript is non-empty: `reply = ask_llm(transcript, system=SYSTEM_PROMPT)`.  
   - Reply is printed (`→ TONY: Hey bro! …`).

5. **1.3 TTS**  
   - `play_tts_reply(reply)` → `text_to_speech(reply)` → Kokoro generates audio → `sd.play(audio, sr)` then `sd.wait()`.

6. **Loop**  
   - Buffer is reset; the loop continues and waits for the next utterance.

So: **1.1 → 1.2 → 1.3** are chained in order in `live_transcript.py`; no separate services, all in one process except Ollama (which is a separate server).

---

## 6. CUDA vs CPU: What we use and why

### Short summary

| Component | Device we use | Where it’s set | Reason |
|-----------|----------------|----------------|--------|
| **1.1 STT (Faster Whisper)** | **GPU (CUDA)** | `live_transcript.py`: `DEVICE = "cuda"` | Faster; needs CUDA 12 DLLs (cu121). |
| **1.2 LLM (Ollama)**        | **GPU** (typical) | Ollama server config | Our code doesn’t set it; Ollama uses GPU if available. |
| **1.3 TTS (Kokoro)**        | **CPU** (default) | `tts.py`: default `device = "cpu"`; env `TTS_DEVICE` | Avoids cuDNN Error 127 on this Windows setup. |

### Details

- **CUDA:** NVIDIA GPU acceleration. Needs matching driver, CUDA toolkit/cuDNN (or PyTorch’s bundled DLLs).  
- **CPU:** No GPU; slower but no CUDA/cuDNN issues.

**Why STT is on GPU:**  
Faster Whisper (CTranslate2) is built for CUDA 12 on this project. We use PyTorch **cu121**; its `torch/lib` (and the early DLL path fix in `live_transcript.py`) provides `cublas64_12.dll` etc., so STT runs on GPU.

**Why TTS is on CPU by default:**  
Kokoro uses PyTorch. When we set `device="cuda"` for Kokoro, loading cuDNN sometimes fails with “Could not load symbol cudnnGetLibConfig. Error code 127” on this Windows machine. So we default TTS to CPU. You can try GPU with `TTS_DEVICE=cuda`.

**Why we don’t use cu118 for this project:**  
Faster Whisper expects **CUDA 12** DLLs (e.g. `cublas64_12.dll`). PyTorch **cu118** only provides CUDA 11 DLLs, which leads to “Library cublas64_12.dll is not found”. So we keep **cu121** and run TTS on CPU.

---

## 7. Other important points

### Main file and entrypoint

- **Run the bot:** `python live_transcript.py` from the `AI_Bot` folder (with the right venv activated).
- **Main script:** `live_transcript.py` — ties together mic, 1.1, 1.2, and 1.3.

### Config and environment

- **STT:** `MODEL_SIZE`, `DEVICE`, `SAMPLE_RATE`, `LANGUAGE`, `SILENCE_DURATION_SEC`, etc. in `live_transcript.py`.
- **LLM:** `OLLAMA_MODEL`, `SYSTEM_PROMPT` in `llm.py`.
- **TTS:** `TTS_DEVICE` (env), `DEFAULT_VOICE`, `DEFAULT_LANG`, `ENABLE_TTS` in `tts.py` / `live_transcript.py`.

### Test / standalone scripts

- **test_llm.py** — Text-only LLM test (no STT/TTS).
- **test_tts.py** — TTS-only test (generates and plays one WAV).
- **test_whisper.py** — If present, for STT-only checks.

These are for development; the main flow is `live_transcript.py`.

### Windows-specific behavior

- At the top of `live_transcript.py`, on Windows we add PyTorch’s `torch/lib` to the DLL search path so CUDA 12 DLLs are found for Faster Whisper.
- In `tts.py`, when `TTS_DEVICE != "cpu"` we also try to add that path before Kokoro loads (to reduce cuDNN issues if you try TTS on GPU).

### Known issue and workaround

- **Issue:** With TTS on GPU (`TTS_DEVICE=cuda`), you may see “Could not load symbol cudnnGetLibConfig. Error code 127”.
- **Workaround:** Keep TTS on CPU (default). No code change needed; only set `TTS_DEVICE=cuda` if you explicitly want to try GPU.

### Dependencies (high level)

- **1.1:** faster_whisper, sounddevice, numpy (and PyTorch cu121 for CUDA DLLs).
- **1.2:** ollama (Python); Ollama server installed and running.
- **1.3:** kokoro, soundfile, PyTorch; sounddevice for playback.

---

## 8. Quick reference

| Part | Name    | Input    | Output   | Device (current) |
|------|---------|----------|----------|------------------|
| 1.1  | STT     | Mic audio| Text     | GPU (cuda)       |
| 1.2  | LLM     | Text     | Text     | GPU (Ollama)     |
| 1.3  | TTS     | Text     | Audio    | CPU (default)    |

**Flow:** Mic → STT → transcript → LLM → reply → TTS → play → (loop).

**Document version:** Reflects the codebase as of this write-up; if you change `DEVICE`, `TTS_DEVICE`, or Ollama config, update this doc accordingly.
