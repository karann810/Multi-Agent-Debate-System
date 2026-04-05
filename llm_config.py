"""
llm_config.py
─────────────
Single place to configure the LLM.
All agents and nodes import from here.
"""

import os
from langchain_groq import ChatGroq


def get_llm(temperature: float = 0.7) -> ChatGroq:
    """
    Returns a ChatGroq instance.
    Each agent calls this with their own temperature.

    Model: llama-3.3-70b-versatile  (fast, free tier friendly)
    Swap to: mixtral-8x7b-32768 / gemma2-9b-it if needed
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "\n❌ GROQ_API_KEY not set.\n"
            "   Get your free key at: https://console.groq.com\n"
            "   Then run: export GROQ_API_KEY='your-key-here'\n"
        )

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=api_key,
        max_tokens=1024,
    )

# temp for each agent so as to get perfect response
# Higher = more creative/unpredictable
# Lower  = more precise/consistent/predictable

AGENT_TEMPERATURES = {
    "optimist":        0.8,
    "pessimist":       0.4,
    "devils_advocate": 0.9,
    "pragmatist":      0.3,
    "orchestrator":    0.2,   # needs to be analytical
    "synthesizer":     0.2,   # needs structured, consistent output
}
