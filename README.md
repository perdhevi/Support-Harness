# SupportHarness

> A teaching repo for building an agent harness one feature at a time. Each
> stage adds exactly one harness-level concern on top of the previous one,
> with the ReAct loop itself staying as close to identical as possible.
>
> Runs locally. Ollama + llama3.2. No API keys.

---

## Where we are right now

| Stage | What it adds | Status | File |
|---|---|---|---|
| **1** | Bare ReAct loop with two KB tools (`search_kb`, `get_article`) | ✅ Complete | [`stage1_bare_react.py`](stage1_bare_react.py) · [readme](README_stage1.md) |
| **2** | `AGENTS.md` injection + ChromaDB episodic memory with cosine-distance relevance threshold (`recall_memory`, `save_to_memory`) | ✅ Complete | [`stage2_memory.py`](stage2_memory.py) · [readme](README_stage2.md) |
| **3** | `TokenTracker` + tool output trimming + budget guard | ✅ Complete | [`stage3_tokens.py`](stage3_tokens.py) · [readme](README_stage3.md) |
| **4** | LLM-as-judge quality gate (helpfulness + groundedness scores, bounded retries) | ✅ Complete | [`stage4_critique.py`](stage4_critique.py) · [readme](README_stage4.md) |
| **5+** | Open: relevance-based retention, parallel tool calls, evals harness, multi-model judge | 📋 Backlog | — |

Every stage's REPL has an in-app `guide` command that walks through a focused
demo of what that stage adds — type `guide` after launch.

---

## The progression diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                           SHARED CORE                                │
│                                                                      │
│  articles.json  ──►  mock_kb.py  ──►  search_kb / get_article        │
│       (data)            (logic)              (tools)                 │
└──────────────────────────────────────────────────────────────────────┘

     Stage 1            Stage 2              Stage 3              Stage 4
 ┌────────────┐    ┌──────────────┐    ┌────────────────┐    ┌────────────────┐
 │ ReAct loop │ ─► │ + AGENTS.md  │ ─► │ + TokenTracker │ ─► │ + LLM-as-judge │
 │ driven by  │    │ + ChromaDB   │    │ + tool trimming│    │ + retry loop   │
 │finish_reason│    │ + recall_mem │    │ + budget guard │    │ + quality gate │
 └────────────┘    └──────────────┘    └────────────────┘    └────────────────┘
  "model+tools"    "memory across     "context-rot         "answer-quality
                    runs"              proof"                proof"
```

Each stage's file is **self-contained** — you can run any stage on its own
without touching the others. The only thing they share is `mock_kb.py` and
`articles.json`.

---

## Quick start

```bash
# one-time setup
ollama pull llama3.2
ollama serve            # skip if already running
pip install openai chromadb

# run any stage
python stage1_bare_react.py     # bare loop
python stage2_memory.py         # + memory
python stage3_tokens.py         # + token tracking & trimming
python stage4_critique.py       # + LLM-as-judge quality gate

# wipe stored memory for a clean demo run
python reset_memory.py          # all stages (with y/N prompt)
python reset_memory.py 2 4      # only stages 2 and 4
python reset_memory.py -y       # no prompt
```

Each stage opens an interactive REPL. Type `guide` for a built-in demo
walkthrough, `quit` to exit. Each stage's README has a list of suggested
questions and experiments.

---

## What each stage teaches

### Stage 1 — Bare ReAct
The simplest possible agent harness: model + tools + loop. The loop is
driven entirely by `finish_reason`. Every turn reprints the full
conversation history — context bloat is *visible*, but not handled.

> **Pain point exposed:** memory gap. The agent forgets every conversation
> the moment `run_agent()` returns.

### Stage 2 — AGENTS.md + ChromaDB Memory
The harness injects a behavioral guide (`AGENTS_MD`) the model can't
modify, and adds a `recall_memory` tool plus an automatic
`save_to_memory` step after every run. The agent now has access to its
own past resolutions across sessions.

> **Pain point exposed:** context rot. Memory works, but each run can
> still drag in a 2 KB article body, then another, then another — every
> one re-billed every turn.

### Stage 3 — TokenTracker + Tool Output Trimming
The harness now accounts for tokens explicitly: a `[budget]` bar before
each model call, per-turn + cumulative usage after each answer. Older
tool message bodies are auto-replaced with stubs. The conversation can
no longer silently fill the context window.

> **Pain point exposed:** quality drift. The agent uses the right
> articles and stays inside its budget, but a confidently-wrong answer
> still ships unchallenged.

### Stage 4 — LLM-as-Judge Quality Gate
Before the harness returns the model's final answer to the user, it runs
a SECOND model call in `judge` mode. The judge sees the question, the
tool transcript, and the candidate answer, and emits a strict JSON
verdict with `helpfulness` and `groundedness` scores (1-5). On `revise`,
the harness injects the critique back into `messages` and re-runs the
ReAct loop, up to `MAX_RETRIES` times. If all attempts fail, the best
draft ships with a clearly labeled quality warning.

> **What's now solved at the answer layer:** "looks confident, isn't grounded"
> drafts get blocked or rewritten. An honest "the KB doesn't cover this" is
> explicitly defined as a passing answer — the gate exists to block
> hallucination, not to bully the model into false confidence.
>
> **Pain point exposed (next):** no regression evals. Every gate decision
> is currently anecdotal. We don't yet measure whether Stage 4 actually
> raises pass-rates against a fixed question set vs. Stage 3.

---

## Repo layout

```
support-harness/
├── README.md                  ← you are here
├── articles.json              ← KB data (19 articles, 5 products, 3 internal)
├── synonyms.json              ← lightweight query expansion data
├── mock_kb.py                 ← shared KB logic (search_kb / get_article)
├── agent.md                   ← system prompt used by Stage 1
├── agent_investigation.md     ← richer behavioural guide used by Stages 2-4
├── reset_memory.py            ← wipe one or more .chroma_stageN/ dirs
│
├── stage1_bare_react.py       ← Stage 1 implementation
├── README_stage1.md
│
├── stage2_memory.py           ← Stage 2 implementation
├── README_stage2.md
│   └── .chroma_stage2/        ← created on first run
│
├── stage3_tokens.py           ← Stage 3 implementation
├── README_stage3.md
│   └── .chroma_stage3/        ← created on first run
│
└── stage4_critique.py         ← Stage 4 implementation
    ├── README_stage4.md
    └── .chroma_stage4/        ← created on first run
```

`articles.json` and `mock_kb.py` are intentionally separate. The JSON is
pure data anyone can edit without touching Python. The Python is pure
logic that never hardcodes article content.

Each stage uses its own ChromaDB path so memory from one stage never
contaminates an A/B comparison against another.

---

## Roadmap — what comes after Stage 4

The four canonical harness concerns are now all covered:

| Stage | Concern              |
|-------|----------------------|
| 1     | Loop control         |
| 2     | Memory + behaviour   |
| 3     | Context budget       |
| 4     | Output quality       |

Beyond Stage 4 the work is no longer about plugging missing primitives —
it's about optimisation and measurement:

- **Evals harness.** A fixed question set with golden answers and a
  pass-rate metric across stages. Without this, every claim about
  Stage 4 raising quality is anecdotal.
- **Multi-model judge.** Run the gate against a different model than the
  agent (smaller for cost, or stronger for accuracy). Independence makes
  the gate harder to game.
- **Relevance-based retention.** Stage 3 trims by recency. A high-value
  early article can lose to a low-value recent one. Score relevance and
  trim the lowest-relevance bodies first.
- **Parallel tool dispatch.** The dispatcher today processes `tool_calls`
  serially. Run independent calls in parallel for latency.
- **Streaming + early termination.** Stop generation when the model is
  visibly looping rather than waiting for `MAX_TURNS`.

---

## Design principles

1. **One concern per stage.** Each stage adds exactly one feature so the
   diff is the lesson.
2. **Harness vs. model.** The harness owns AGENTS.md, memory persistence,
   token tracking, trimming, and quality gating. The model only ever sees
   tools and a system prompt.
3. **Loop logic stays stable.** The ReAct loop is essentially the same in
   all four stages. New behavior is layered around it, not threaded
   through it. (Stage 4 factors it into `_react_loop()` so the outer gate
   loop can call it more than once — same logic, just reusable.)
4. **Local-first.** Ollama + llama3.2 means no keys, no quotas, no excuse
   not to run the experiments.
5. **Knobs, not magic.** Every threshold (`MAX_TURNS`, `WARN_FRAC`,
   `KEEP_RECENT_TOOL_OUTPUTS`, `PASS_THRESHOLD`, `MAX_RETRIES`, etc.) is a
   constant at the top of the file. Tweak and re-run.
6. **Fail loudly.** Stage 3 hard-stops on budget overrun. Stage 4 ships a
   tagged warning rather than a silent bad answer. The harness should
   never degrade quietly.
