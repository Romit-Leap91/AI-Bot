# Whisper-Live Setup (Fixed Install)

Whisper-live is installed with **faster-whisper 1.2.1** (no downgrade). The server and most features work. One optional dependency needs a manual step on Windows.

## What’s installed

- **whisper-live** (--no-deps to avoid version conflicts)
- **faster-whisper 1.2.1** (unchanged)
- **torch**, **torchaudio**, **websockets**, **scipy**, **websocket-client**, **numba**, **soundfile**, **librosa**

## Running the server (faster_whisper backend)

```powershell
cd C:\Users\LENOVO\Documents\Voice_Bot\AI_Bot
.venv\Scripts\activate
python -m whisper_live.server --port 9090 --backend faster_whisper
```

Or use the repo’s `run_server.py` if you cloned WhisperLive.

Use **device CPU** if you don’t have CUDA (to avoid `cublas64_12.dll` errors), e.g. in code or via env.

## PyAudio (needed only for the client mic / playback)

The **whisper-live client** (microphone input and file playback) requires **PyAudio**. On Windows with Python 3.14 there is no pre-built wheel, so it must be built from source.

1. Install **Microsoft C++ Build Tools** (Visual Studio Build Tools):
   - https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Run the installer and select “Desktop development with C++”.

2. Restart your terminal, then:

   ```powershell
   .venv\Scripts\pip.exe install PyAudio
   ```

After that, `from whisper_live.client import TranscriptionClient` and mic/file client usage will work.

## If you only use the server

If you only run the **WhisperLive server** (e.g. with faster_whisper) and connect from another client (browser, another app), you don’t need PyAudio. The server imports and runs without it.
