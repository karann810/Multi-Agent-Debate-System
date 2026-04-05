from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

from llm_config import get_llm, AGENT_TEMPERATURES
from state import DebateState

# Persona System Prompts

PERSONAS = {
    "optimist": """You are the OPTIMIST analyst in a structured debate panel.

Your role:
- Find genuine opportunities, upside potential, and reasons why something will succeed
- Acknowledge risks but reframe them as challenges that CAN be solved
- Use real-world examples of similar things that succeeded
- Be enthusiastic but not naive — back your optimism with logic and evidence

Your debate style:
- Lead with the strongest possible case FOR the idea
- When challenged, defend your position with new angles, not just repetition
- Keep responses to 3-5 focused paragraphs
- Be assertive and direct. Make a clear case.""",

    "pessimist": """You are the PESSIMIST analyst in a structured debate panel.

Your role:
- Identify what will realistically go wrong and what's being overlooked
- Challenge assumptions that others take for granted
- Demand evidence — vague optimism is not a strategy
- Be rigorous and precise, not just contrarian

Your debate style:
- Lead with the most critical flaw in the proposal
- Hold your ground with data or logic when challenged
- Keep responses to 3-5 focused paragraphs
- Don't soften everything. Make people uncomfortable with hard truths.""",

    "devils_advocate": """You are the DEVIL'S ADVOCATE in a structured debate panel.

Your role:
- Take the opposite position of whoever is currently winning the argument
- If optimism dominates, attack it. If pessimism dominates, defend the idea.
- Expose logical gaps, false dichotomies, and hidden assumptions in ALL arguments
- Your goal is to prevent groupthink, not to hold a fixed personal view

Your debate style:
- Identify the strongest argument and attack its foundation
- Ask the question nobody else is asking
- Shift the frame entirely when the debate gets stuck
- Keep responses to 3-5 focused paragraphs. Be provocative and precise.""",

    "pragmatist": """You are the PRAGMATIST analyst in a structured debate panel.

Your role:
- Cut through theory and focus on what is actually doable given real constraints
- Consider: budget, time, team size, market reality, execution difficulty
- Ask "what would you actually DO on Monday morning?"
- Bridge between ideal and achievable

Your debate style:
- Ground every abstract argument in concrete reality
- Call out when the debate is too theoretical to be useful
- Provide the specific, actionable version of what others propose
- Keep responses to 3-5 focused paragraphs. Be concrete and direct."""
}


# Helper: Build debate context string for Round 2+

def _build_previous_rounds_context(state: DebateState, current_agent: str) -> str:
   # in this function we will build summary/context of all previous rounds responses of all agents
    if state["current_round"] == 1:
        return ""  # Round 1 — nobody has spoken yet

    agent_map = {
        "optimist":        (" Optimist",        state["optimist_responses"]),
        "pessimist":       (" Pessimist",        state["pessimist_responses"]),
        "devils_advocate": (" Devil's Advocate", state["devils_advocate_responses"]),
        "pragmatist":      (" Pragmatist",       state["pragmatist_responses"]),
    }

    context =  "\n---- DEBATE HISTORY SO FAR ----\n" 

    # Show each previous round
    for round_num in range(1, state["current_round"]):
        context += f"\n\n── ROUND {round_num} ──\n"
        for agent_name, (display, responses) in agent_map.items():
            if len(responses) >= round_num:
                label = "(YOUR previous response)" if agent_name == current_agent else ""
                context += f"\n{display} {label}:\n{responses[round_num - 1]}\n"

    return context


# Helper: Build round-specific instruction

def _get_round_instruction(round_number: int, max_rounds: int) -> str:
    if round_number == 1:
        return "This is Round 1. Give your opening position. Be direct and set up your strongest argument."
    
    elif round_number == max_rounds:
        return "This is the FINAL round. Summarize your core position. Acknowledge the strongest point made against you. Be decisive."
    
    else:
        return f"This is Round {round_number}. You've seen everyone's arguments. Engage directly with the strongest point against your position. Advance the debate — don't just repeat yourself."

# Agent Node Factory
# Creates a LangGraph node function for any agent

def make_agent_node(agent_name: str):
   
    # Usage:
    #     optimist_node = make_agent_node("optimist")
    #     # Then add to graph: graph.add_node("optimist", optimist_node)
  

    # Map agent names to their state response list keys
    response_key_map = {
        "optimist":        "optimist_responses",
        "pessimist":       "pessimist_responses",
        "devils_advocate": "devils_advocate_responses",
        "pragmatist":      "pragmatist_responses",
    }

    display_names = {
        "optimist":        " Optimist",
        "pessimist":       " Pessimist",
        "devils_advocate": " Devil's Advocate",
        "pragmatist":      " Pragmatist",
    }

    persona = PERSONAS[agent_name]
    temperature = AGENT_TEMPERATURES[agent_name]
    response_key = response_key_map[agent_name]
    display_name = display_names[agent_name]

   
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{persona}"),
        ("human", "{user_message}"),
    ])
    llm = get_llm(temperature=temperature)
    parser = StrOutputParser()
    chain = prompt | llm | parser

    def node(state: DebateState) -> dict:
        topic = state["topic"]
        round_num = state["current_round"]
        note = state["orchestrator_notes"].get(agent_name, "")

        history_context = _build_previous_rounds_context(state, agent_name)
        round_instruction = _get_round_instruction(round_num, max_rounds=state["max_rounds"] )

        if round_num == 1:
            user_message = f"""DEBATE TOPIC: {topic}

{round_instruction}"""
        else:
            user_message = f"""DEBATE TOPIC: {topic}

{history_context}

{round_instruction}"""

        # Add orchestrator injection if present
        if note:
            user_message += f"\n\n⚠️ MODERATOR NOTE: {note}"

        print(f"\n  {display_name} responding (Round {round_num})...")

        # Invoke the chain
        response = chain.invoke({
            "persona": persona,
            "user_message": user_message,
        })

        print(f"\n{display_name}:")
        print("─" * 45)
        print(response)

        # Return state update
        # LangGraph will APPEND this to the existing list (operator.add)
        return {
            response_key: [response],
            "all_responses": [{
                "round": round_num,
                "agent": agent_name,
                "display": display_name,
                "response": response,
            }]
        }

    # Name the function properly for LangGraph graph visualization
    node.__name__ = f"{agent_name}_node"
    return node


# Create all 4 agent nodes

optimist_node        = make_agent_node("optimist")
pessimist_node       = make_agent_node("pessimist")
devils_advocate_node = make_agent_node("devils_advocate")
pragmatist_node      = make_agent_node("pragmatist")
