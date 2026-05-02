"""
Shared model configuration for all stages.
Change MODEL here to switch every stage at once.
"""

from openai import OpenAI

MODEL = "gemma4"

# Ollama's OpenAI-compatible endpoint.
# api_key must be non-empty but the value doesn't matter.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
