

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_config import get_llm, AGENT_TEMPERATURES
from state import DebateState


#  Synthesizer prompts 

SYNTHESIZER_SYSTEM = """You are a senior analyst who just observed a full structured debate 
between four analysts: Optimist, Pessimist, Devil's Advocate, and Pragmatist.

Your job: synthesize the debate into actionable intelligence.
You do NOT pick a winner. You extract the signal from the noise.

You must respond with ONLY valid JSON — no preamble, no markdown fences."""

SYNTHESIZER_HUMAN = """DEBATE TOPIC: {topic}

FULL DEBATE TRANSCRIPT:
{transcript}

Produce ONLY this JSON structure (no markdown, no backticks):
{{
  "verdict": "Clear 1-2 sentence recommendation on what should actually be done",
  "confidence": 72,
  "confidence_reasoning": "One sentence on what makes this uncertain",
  "consensus_points": [
    "Something all agents implicitly agreed on despite arguing differently",
    "Another shared assumption or conclusion"
  ],
  "core_tension": "The one fundamental disagreement that was never fully resolved",
  "minority_view": "The most interesting argument that was underexplored and deserves more thought",
  "key_risk": "The single most important risk to account for",
  "next_action": "Specific, concrete thing to do in the next 7 days",
  "debate_quality": "One sentence on what made this debate useful or where it fell short"
}}"""


# Format transcript from state

def build_transcript(state: DebateState) -> str:
    """
    Build a clean, readable transcript from the all_responses list in state.
    Groups responses by round for readability.
    """
    topic = state["topic"]
    all_responses = state["all_responses"]
    max_rounds = state["max_rounds"]

    transcript = f"""
  DEBATE TRANSCRIPT
  Topic: {topic}
"""

    for round_num in range(1, max_rounds + 1):
        round_responses = [r for r in all_responses if r["round"] == round_num]
        if not round_responses:
            continue

        transcript += f"\n\n{'═' * 55}\n  ROUND {round_num}\n{'═' * 55}"
        for entry in round_responses:
            transcript += f"\n\n{entry['display']}:\n{'─' * 40}\n{entry['response']}"

    return transcript



# Format synthesis output for terminal display


def format_synthesis_output(topic: str, synthesis: dict) -> str:
    """Pretty-print the synthesis dict to terminal."""
    confidence = synthesis.get("confidence", 0)
    filled = int(confidence / 5)
    bar = "█" * filled + "░" * (20 - filled)

    consensus = synthesis.get("consensus_points", [])
    consensus_text = "\n".join(f"  ✓ {p}" for p in consensus) if consensus else "  None identified"

    return f"""
  SYNTHESIS & VERDICT
  Topic: {topic}


📋  VERDICT
{'─' * 60}
{synthesis.get('verdict', 'No verdict produced')}

📊  CONFIDENCE: {confidence}%
     {bar}
     {synthesis.get('confidence_reasoning', '')}

✅  WHAT EVERYONE AGREED ON
{'─' * 60}
{consensus_text}

⚡  CORE TENSION  (never resolved)
{'─' * 60}
{synthesis.get('core_tension', 'None identified')}

🔍  MINORITY VIEW  (worth preserving)
{'─' * 60}
{synthesis.get('minority_view', 'None')}

⚠️   KEY RISK
{'─' * 60}
{synthesis.get('key_risk', 'None identified')}

🎯  NEXT ACTION  (do this week)
{'─' * 60}
{synthesis.get('next_action', 'No specific action identified')}

💬  DEBATE QUALITY
{'─' * 60}
{synthesis.get('debate_quality', '')}

{'═' * 60}"""


# LangGraph node


def synthesizer_node(state: DebateState) -> dict:
    """
    LangGraph node: runs after all debate rounds complete.
    
    1. Builds the full transcript from state
    2. Calls ChatGroq to produce structured synthesis
    3. Returns state update with synthesis + transcript
    """
    print(f"\n{'═' * 55}")
    print("  SYNTHESIZER — Analyzing full debate...")
    print(f"{'═' * 55}\n")

    topic = state["topic"]
    transcript = build_transcript(state)

    # Build LangChain chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIZER_SYSTEM),
        ("human", SYNTHESIZER_HUMAN),
    ])
    llm = get_llm(temperature=AGENT_TEMPERATURES["synthesizer"])
    parser = StrOutputParser()
    chain = prompt | llm | parser

    #  Invoke 
    raw = chain.invoke({
        "topic": topic,
        "transcript": transcript,
    })

    # Parse JSON 
    try:
        clean = raw.strip()
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    clean = part
                    break

        synthesis = json.loads(clean)

    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠️  Synthesis parse issue: {e}")
        synthesis = {
            "verdict": raw,
            "confidence": 0,
            "confidence_reasoning": "Could not parse structured output",
            "consensus_points": [],
            "core_tension": "Unknown",
            "minority_view": "Unknown",
            "key_risk": "Unknown",
            "next_action": "Review the full transcript manually",
            "debate_quality": "Parse error — see raw output above",
        }

    #  Print formatted output 
    formatted = format_synthesis_output(topic, synthesis)
    print(formatted)

    return {
        "synthesis": synthesis,
        "transcript": transcript,
    }
