

from typing import TypedDict, Annotated
import operator


class DebateState(TypedDict):
    
    # Fields with `Annotated[list, operator.add]` are APPEND-ONLY —
    # LangGraph merges new items into the existing list automatically.
    

    topic: str                              # The debate question

    current_round: int                      # 1, 2, or 3
    max_rounds: int                         # default 3

    optimist_responses:        Annotated[list, operator.add]
    pessimist_responses:       Annotated[list, operator.add]
    devils_advocate_responses: Annotated[list, operator.add]
    pragmatist_responses:      Annotated[list, operator.add]

  
    # Each entry will be in form of -> {round: int, agent: str, response: str}
    all_responses: Annotated[list, operator.add]


    orchestrator_notes: dict

    # ── Final output ────────────────────────────────────────────────────────
    synthesis: dict                         # Structured verdict from synthesizer
    transcript: str                         # Full formatted debate text
    is_complete: bool                       # True when debate is done


def initial_state(topic: str, max_rounds: int = 3) -> DebateState:
    """
    Create a clean starting state for a new debate.
    Called once at the beginning before the graph runs.
    """
    return DebateState(
        topic=topic,
        current_round=1,
        max_rounds=max_rounds,
        optimist_responses=[],
        pessimist_responses=[],
        devils_advocate_responses=[],
        pragmatist_responses=[],
        all_responses=[],
        orchestrator_notes={},
        synthesis={},
        transcript="",
        is_complete=False,
    )
