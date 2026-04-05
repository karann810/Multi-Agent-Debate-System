"""
orchestrator.py
───────────────
The Orchestrator is a LangGraph node that runs BETWEEN rounds.

What it does:
1. Reads the completed round responses from DebateState
2. Uses ChatGroq to analyze: is anyone dodging arguments?
3. Returns targeted injection notes for the NEXT round
4. Also handles: incrementing the round counter

It uses LangChain's JsonOutputParser for structured output.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_config import get_llm, AGENT_TEMPERATURES
from state import DebateState


# ── Orchestrator prompt ────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = """You are a sharp debate moderator. You observe structured debates 
and identify when participants are avoiding strong arguments or being repetitive.

Your job: after each round, decide if any agent needs a targeted instruction 
for the next round to make the debate more rigorous and useful.

You must respond with ONLY valid JSON — no preamble, no explanation, no markdown."""

ORCHESTRATOR_HUMAN = """DEBATE TOPIC: {topic}

ROUND {round_number} JUST COMPLETED. Here are the responses:

{round_summary}

Analyze this round. Identify if any agent is:
1. Dodging a strong argument made by another agent
2. Being too repetitive (just restating their opening)
3. Missing a critical angle that would improve the debate

Return ONLY this JSON (no markdown, no backticks):
{{
  "injections": {{
    "optimist": "specific 1-sentence instruction, or null",
    "pessimist": "specific 1-sentence instruction, or null",
    "devils_advocate": "specific 1-sentence instruction, or null",
    "pragmatist": "specific 1-sentence instruction, or null"
  }},
  "reasoning": "one sentence on what you noticed"
}}

Rules:
- Return null for agents who are debating well
- Keep each instruction under 25 words
- Be specific — name the argument they're avoiding"""


def orchestrator_node(state: DebateState) -> dict:
    """
    LangGraph node: runs after each round completes.
    
    - Increments round counter
    - If more rounds remain, analyzes current round and generates injection notes
    - If debate is complete, sets is_complete = True
    """
    current_round = state["current_round"]
    max_rounds = state["max_rounds"]

    print(f"\n{'═' * 55}")
    print(f"  ORCHESTRATOR — End of Round {current_round}")
    print(f"{'═' * 55}")

    # ── If we've finished all rounds, mark complete ─────────────────────
    if current_round >= max_rounds:
        print("  All rounds complete. Moving to synthesis...")
        return {
            "current_round": current_round + 1,
            "is_complete": True,
            "orchestrator_notes": {},
        }

    # Get just this round's responses
    round_summary = _get_round_summary(state, current_round)

    print(f"  Analyzing Round {current_round} for targeted injections...")

    # Build and invoke the chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", ORCHESTRATOR_SYSTEM),
        ("human", ORCHESTRATOR_HUMAN),
    ])
    llm = get_llm(temperature=AGENT_TEMPERATURES["orchestrator"])
    parser = StrOutputParser()
    chain = prompt | llm | parser

    try:
        raw = chain.invoke({
            "topic": state["topic"],
            "round_number": current_round,
            "round_summary": round_summary,
        })

      
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

        data = json.loads(clean)
        injections = data.get("injections", {})
        reasoning = data.get("reasoning", "")

        # Filter out null values
        active = {k: v for k, v in injections.items() if v and v != "null" and v is not None}

        if active:
            print(f"  Reasoning: {reasoning}")
            for agent, note in active.items():
                print(f"  → {agent}: {note}")
        else:
            print(f"  No injections needed. Round {current_round} was solid.")

        return {
            "current_round": current_round + 1,
            "orchestrator_notes": active,
            "is_complete": False,
        }

    except (json.JSONDecodeError, Exception) as e:
        print(f"  Orchestrator parse issue ({e}) — continuing without injections")
        return {
            "current_round": current_round + 1,
            "orchestrator_notes": {},
            "is_complete": False,
        }


def _get_round_summary(state: DebateState, round_num: int) -> str:
    """Extract just this round's responses as a readable string."""
    agent_keys = {
        "optimist":        ("@ Optimist",         state["optimist_responses"]),
        "pessimist":       ("@ Pessimist",        state["pessimist_responses"]),
        "devils_advocate": ("@ Devil's Advocate", state["devils_advocate_responses"]),
        "pragmatist":      ("@ Pragmatist",       state["pragmatist_responses"]),
    }

    summary = ""
    for agent_name, (display, responses) in agent_keys.items():
        if len(responses) >= round_num:
            # Truncate to 400 chars for the analysis prompt
            snippet = responses[round_num - 1][:400] + "..."
            summary += f"\n[{display}]:\n{snippet}\n"

    return summary
