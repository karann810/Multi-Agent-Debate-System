# 🎭 Multi-Agent Debate System
### Built with LangChain + LangGraph + ChatGroq

A structured debate framework where 4 AI agents with distinct personalities 
argue any topic across multiple rounds — facilitated by an orchestrator and 
concluded by a synthesizer that produces an actionable verdict.

---

## Architecture

```
START
  │
  ▼
[optimist] → [pessimist] → [devils_advocate] → [pragmatist]
                                                      │
                                               [orchestrator]
                                                  │       │
                                    more rounds ◄─┘       └─► [synthesizer] → END
```

Built with **LangGraph StateGraph** — the state flows through every node 
and gets updated immutably at each step.

---

## The 4 Agents

| Agent | Personality | Temperature |
|-------|-------------|-------------|
| 🟢 Optimist | Finds opportunity and upside, backs claims with evidence | 0.8 |
| 🔴 Pessimist | Finds risks and hard truths, demands evidence | 0.4 |
| 🟡 Devil's Advocate | Challenges whoever is winning, prevents groupthink | 0.9 |
| 🔵 Pragmatist | Grounds everything in real constraints and numbers | 0.3 |

Each agent is a **LangChain chain**: `ChatPromptTemplate | ChatGroq | StrOutputParser`

---

## How Information Flows

```
Round 1: Each agent sees → topic only
Round 2: Each agent sees → topic + all Round 1 responses + optional orchestrator note
Round 3: Each agent sees → topic + all Round 1 + Round 2 responses + optional note

Synthesizer sees → complete transcript of all 3 rounds
```

The **Orchestrator** analyzes each completed round and injects targeted pressure:
```
"Optimist — you haven't addressed the Pragmatist's point about runway. Do so now."
```

---

## Project Structure

```
multi_agent_debate_v2/
│
├── main.py           ← Entry point. Run this.
├── graph.py          ← LangGraph StateGraph definition
├── state.py          ← DebateState TypedDict (shared data structure)
├── agents.py         ← 4 agent nodes (LangChain chains)
├── orchestrator.py   ← Analyzes rounds, injects notes, routes flow
├── synthesizer.py    ← Final verdict node
├── llm_config.py     ← ChatGroq setup + temperature presets
└── requirements.txt
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free API key from Groq
#    https://console.groq.com

# 3. Set the key
export GROQ_API_KEY='your-key-here'

# 4. Run — interactive mode
python main.py

# 5. Or pass topic directly
python main.py "Should I quit my job and start an AI startup?"

# 6. Custom number of rounds
python main.py "Should we use microservices?" --rounds 2
```

---

## Example Output

```
════════════════════════════════════════════════════════
  ROUND 1
════════════════════════════════════════════════════════

🟢 Optimist:
There has never been a better time to build in AI...

🔴 Pessimist:
The failure rate for AI startups is catastrophic...

🟡 Devil's Advocate:
Both of you are missing the actual question here...

🔵 Pragmatist:
Let's talk numbers. What's your monthly burn rate?...

════════════════════════════════════════════════════════
  SYNTHESIS & VERDICT
════════════════════════════════════════════════════════

📋 VERDICT
Don't quit yet. Spend 90 days validating evenings/weekends.
Set a trigger: if you hit $2k MRR, then quit.

📊 CONFIDENCE: 74%
     ██████████████░░░░░░
     Uncertainty around personal risk tolerance and idea quality.

✅ WHAT EVERYONE AGREED ON
  ✓ The AI opportunity window is real
  ✓ Quitting without validation is reckless

⚡ CORE TENSION
  Opportunity cost of staying employed vs. risk of premature quit

🎯 NEXT ACTION
  Define your validation metric today. Write it down.
```

---

## Key LangGraph Concepts Used

**StateGraph** — Graph where each node reads/writes a shared typed state dict

**Annotated[list, operator.add]** — State fields that APPEND across nodes instead of overwriting

**Conditional edges** — `route_after_orchestrator()` decides: loop back or go to synthesizer

**Node functions** — Each node is a plain Python function `(state) → dict`

---

## Cost

Groq has a **free tier** with generous rate limits.  
llama-3.3-70b-versatile: Free on Groq's free plan.  
Each debate = ~13 LLM calls, typically completes in 30-60 seconds.

---
