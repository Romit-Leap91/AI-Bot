"""Ollama/Qwen LLM for voice bot. Used by live_transcript and test_llm."""
import ollama

OLLAMA_MODEL = "ministral-3:3b"  # or "qwen3", "qwen2.5:7b" for lower latency
SYSTEM_PROMPT = "You are TONY, a helpful voice assistant. Reply in one or two short sentences."


def ask_llm(prompt: str, system: str | None = None) -> str:
    """Send user text to Ollama and return the assistant reply."""
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
    )
    return response["message"]["content"].strip()
