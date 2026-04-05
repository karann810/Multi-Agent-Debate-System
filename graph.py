from langgraph.graph import StateGraph, START, END

from state import DebateState
from agents import (
    optimist_node,
    pessimist_node,
    devils_advocate_node,
    pragmatist_node,
)
from orchestrator import orchestrator_node
from synthesizer import synthesizer_node



def route_after_orchestrator(state: DebateState) -> str:
    
    # - If debate is complete → go to synthesizer
    # - If more rounds remain → go back to optimist (start next round)

    if state["is_complete"]:
        return "synthesizer"
    else:
        return "optimist"  # Start the next round



def build_debate_graph() -> StateGraph:
  

    graph = StateGraph(DebateState)

    graph.add_node("optimist",        optimist_node)
    graph.add_node("pessimist",       pessimist_node)
    graph.add_node("devils_advocate", devils_advocate_node)
    graph.add_node("pragmatist",      pragmatist_node)
    graph.add_node("orchestrator",    orchestrator_node)
    graph.add_node("synthesizer",     synthesizer_node)


    graph.add_edge(START, "optimist")

    graph.add_edge("optimist",        "pessimist")
    graph.add_edge("pessimist",       "devils_advocate")
    graph.add_edge("devils_advocate", "pragmatist")
    graph.add_edge("pragmatist",      "orchestrator")


    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator
    )

    graph.add_edge("synthesizer", END)
    
    return graph.compile()
