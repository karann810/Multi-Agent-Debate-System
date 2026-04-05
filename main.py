

import sys
import os
from dotenv import load_dotenv

load_dotenv()

import argparse
from datetime import datetime

from state import initial_state
from graph import build_debate_graph


# Banner

BANNER = """
                  MULTI-AGENT DEBATE SYSTEM
  Built with LangChain + LangGraph + ChatGroq (Llama 3.3 70B)

    Agents:  @ Optimist  @ Pessimist  @ Devil's Advocate  @ Pragmatist
  Flow:    3 rounds → Orchestrator analysis → Synthesizer verdict
"""

EXAMPLES = [
    "Should I quit my job and start an AI startup in 2025?",
    "Should our team build in-house ML infra or use APIs?",
    "Is it worth migrating from a monolith to microservices now?",
    "Should we raise VC funding or stay bootstrapped?",
    "Should I learn Rust or stick with Python for backend work?",
]

# using the below function we will save final results to the txt file

def save_results(final_state: dict, topic: str):
    """Save full transcript + synthesis to a timestamped text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() or c in " _-" else "" for c in topic[:40])
    safe_topic = safe_topic.strip().replace(" ", "_")
    filename = f"debate_{safe_topic}_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_state.get("transcript", ""))
        f.write("\n\n")

        synthesis = final_state.get("synthesis", {})
        if synthesis:
            f.write("═" * 60 + "\n")
            f.write("SYNTHESIS\n")
            f.write("═" * 60 + "\n\n")
            for key, value in synthesis.items():
                if isinstance(value, list):
                    f.write(f"{key.upper()}:\n")
                    for item in value:
                        f.write(f"  • {item}\n")
                else:
                    f.write(f"{key.upper()}:\n{value}\n")
                f.write("\n")

    print(f"\n💾 Full debate saved to: {filename}")
    return filename


# Main

def main():
    print(BANNER)
    if not os.environ.get("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not set.")
        print("   Get your free key at: https://console.groq.com")
        print("   Then run: export GROQ_API_KEY='your-key-here'")
        sys.exit(1)

    #  Parse arguments 
    parser = argparse.ArgumentParser(description="Multi-Agent Debate System")
    parser.add_argument(
        "topic",
        nargs="*",
        help="Debate topic (or leave empty for interactive mode)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of debate rounds (default: 3)"
    )
    args = parser.parse_args()

    #Get topic
    if args.topic:
        topic = " ".join(args.topic)
    else:
        print("Example topics:")
        for i, ex in enumerate(EXAMPLES, 1):
            print(f"  {i}. {ex}")
        print()
        topic = input("Enter your debate topic: ").strip()
        if not topic:
            topic = EXAMPLES[0]
            print(f"\nUsing example: {topic}")

    num_rounds = max(1, min(args.rounds, 5))  # clamp between 1 and 5

    print(f"\n  Topic:  {topic}")
    print(f"  Rounds: {num_rounds}")
    print(f"  LLM:    ChatGroq (llama-3.3-70b-versatile)")
    print("\n  Starting debate...\n")

    # Build graph + run 
    graph = build_debate_graph()
    state = initial_state(topic=topic, max_rounds=num_rounds)

    final_state = graph.invoke(state)

    save_results(final_state, topic)

    print("\n✅ Debate complete!")


if __name__ == "__main__":
    main()
