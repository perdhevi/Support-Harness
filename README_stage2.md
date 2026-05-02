# Stage 2 — AGENTS.md Injection + ChromaDB Episodic Memory

## What this stage adds

Stage 2 layers two new harness features on top of the bare Stage 1 loop.
The ReAct loop itself is unchanged — every new line of code is a harness-level
concern, not a model-level one.

```
Stage 1 loop (unchanged)
        +
┌───────────────────────────────────────────────────────────┐
│  NEW: AGENTS.md                                           │
│  Injected as the system prompt prefix on every call.      │
│  The model receives behavioral rules it cannot modify.    │
└───────────────────────────────────────────────────────────┘
        +
┌───────────────────────────────────────────────────────────┐
│  NEW: ChromaDB episodic memory                            │
│                                                           │
│  recall_memory(query) → tool the model calls first        │
│  save_to_memory(q, a) → harness calls after loop ends     │
│                          (invisible to the model)         │
└───────────────────────────────────────────────────────────┘
```

---

## What the model sees now

The system prompt the model receives on every call:

```
# Support Investigation Agent — Behavioral Guide

## Role
You are a technical support investigation agent...

## Investigation rules
1. Always call `recall_memory` first...
2. If memory has a relevant past case, cite it...
3. Call `search_kb` before concluding...
...

## Tone
Concise and direct...
```

This is `AGENTS_MD` in `stage2_memory.py`. The harness injects it — the model
cannot edit it, skip it, or contradict it (it can try, but the harness re-injects
it fresh on every call).

---

## How memory works

```
run_agent("How do I reset my password?")
│
├── Turn 1: model calls recall_memory("How do I reset my password?")
│           harness routes → ChromaDB query
│           result: "No past cases in memory yet."
│
├── Turn 2: model calls search_kb("password reset")
│           ...
│
└── Turn N: finish_reason=stop → answer returned
            harness calls save_to_memory(query, answer)  ← invisible to model
            ChromaDB now has 1 document

--- second run of the same question ---

run_agent("How do I reset my password?")
│
├── Turn 1: model calls recall_memory("How do I reset my password?")
│           result: "[Case 1] Query: How do I reset my password?\n
│                   Resolution: To reset your password..."
│
└── Turn 2: finish_reason=stop  ← model skipped KB, cited memory directly
```

The memory gap from Stage 1 is gone. The agent now has access to its own
history.

---

## File layout

```
support-harness/
├── articles.json           # KB data (unchanged)
├── mock_kb.py              # KB logic (unchanged)
├── stage1_bare_react.py    # Stage 1 (unchanged reference)
└── stage2_memory.py        # THIS STAGE
    └── .chroma_stage2/     # created on first run
```

Stage 2 uses a separate ChromaDB path (`.chroma_stage2/`) so it doesn't
pollute Stage 1 or your real KB collection.

---

## Setup

```bash
pip install openai chromadb
ollama pull llama3.2
ollama serve   # skip if already running
```

---

## Run it

```bash
python stage2_memory.py
```

---

## Demonstration: memory in action

Run these four questions in order in a single session. Each one reveals a
different aspect of how the memory layer behaves.

---

### Question 1 — cold start (memory is empty)

```
You › How do I reset my password?
```

What happens internally:

```
Turn 1  model calls  recall_memory("How do I reset my password?")
        harness      → ChromaDB query
        result       "No past cases in memory yet."

Turn 2  model calls  search_kb(query="password reset", product=None)
        result       [kb-001] How to reset your password

Turn 3  model calls  get_article(article_id="kb-001")
        result       "To reset your password: 1. Click 'Forgot Password'..."

Turn 4  finish_reason=stop
        → Answer returned to user

[harness] Answer saved to memory (1 total)   ← invisible to model
```

**What to notice:** `recall_memory` ran first (rule 1 in AGENTS.md), found
nothing, so the agent fell through to a normal KB search.

---

### Question 2 — exact recall (same question again)

```
You › How do I reset my password?
```

What happens internally:

```
Turn 1  model calls  recall_memory("How do I reset my password?")
        result       [Case 1] Query: How do I reset my password?
                     Resolution: To reset your password, click 'Forgot
                     Password' on the login page...

Turn 2  finish_reason=stop
        → Answer returned, citing Case 1

[harness] Answer saved to memory (2 total)
```

**What to notice:** The agent answered in 2 turns instead of 4. It found an
exact match in memory and cited it directly — no KB search needed. This is the
core memory payoff.

---

### Question 3 — new topic (memory miss on a different domain)

```
You › What are the API rate limits for the Pro plan?
```

What happens internally:

```
Turn 1  model calls  recall_memory("What are the API rate limits for the Pro plan?")
        result       "No relevant past cases found."
                     (Case 1's cosine distance is above the harness threshold,
                      so it's filtered out before the model sees it)

Turn 2  model calls  search_kb(query="API rate limits", product="api")
        result       [kb-002] API rate limits — Pro and Enterprise plans

Turn 3  model calls  get_article(article_id="kb-002")
        result       "Rate limits by plan: Free: 100 req/min, Pro: 1,000 req/min..."

Turn 4  finish_reason=stop
        → Answer returned

[harness] Answer saved to memory (3 total)
```

**What to notice:** The harness pre-filters memory hits by cosine distance
(`_RELEVANCE_THRESHOLD` in `stage2_memory.py`). Case 1 is too distant to pass,
so the model receives a clean "no match" signal instead of having to judge
relevance itself — and falls through to a normal KB search.

---

### Question 4 — semantic recall (similar but not identical wording)

```
You › I'm getting 429 errors from the API
```

What happens internally:

```
Turn 1  model calls  recall_memory("I'm getting 429 errors from the API")
        result       [Case 3] Query: What are the API rate limits for the Pro plan?
                     Resolution: Pro plan allows 1,000 requests/minute. HTTP 429
                     means the limit is exceeded — implement exponential back-off...

Turn 2  finish_reason=stop
        → Answer returned, citing Case 3 and connecting 429 to rate limits

[harness] Answer saved to memory (4 total)
```

**What to notice:** The question uses completely different words ("429 errors"
vs "rate limits") but ChromaDB's cosine similarity matched the semantic meaning
to Case 3. The agent answered in 2 turns without touching the KB.

---

### Check memory state at any point

Type `memory` at the prompt:

```
You › memory

  ChromaDB: 4 case(s) stored in .chroma_stage2/
```

Type `flush` to wipe all stored cases and start fresh:

```
You › flush

  ChromaDB: flushed 4 case(s) — memory is now empty.
```

---

### Turn count comparison

| Question                          | Without memory | With memory (after 1st run) |
|-----------------------------------|:--------------:|:---------------------------:|
| How do I reset my password?       | 4 turns        | 2 turns                     |
| API rate limits (Pro plan)?       | 4 turns        | 4 turns (new topic)         |
| Getting 429 errors from the API   | 4 turns        | 2 turns (semantic match)    |

Memory saves turns on repeated or semantically similar questions. It has no
effect on genuinely new topics — the agent falls through to a normal KB search.

---

## Session commands

| Input     | What it does                                          |
|-----------|-------------------------------------------------------|
| question  | Runs `run_agent()`, saves result to memory            |
| `guide`   | Prints the 4-step memory walkthrough                  |
| `memory`  | Shows how many cases are stored in ChromaDB           |
| `flush`   | Wipes all stored cases, resets ChromaDB collection    |
| `history` | Lists every question asked this session               |
| `quit`    | Ends the session                                      |

To wipe memory for multiple stages at once (without launching a session),
use the repo-level [`reset_memory.py`](reset_memory.py) script.

---

## Knobs

```python
_RELEVANCE_THRESHOLD = 0.7   # cosine-distance cutoff in recall_memory
```

ChromaDB returns cosine distance from 0 (identical) to 2 (opposite). Hits
above the threshold are filtered out *before* the model sees them, so the
relevance judgement is enforced by the harness rather than left to a
small local model. Lower the value for stricter recall (more "no match"),
raise it for noisier recall.

---

## What's next: Stage 3

Stage 3 adds `TokenTracker` + tool output trimming — context rot prevention.
The agent now has memory, but a long session with large KB articles can fill
the context window and silently degrade answer quality. Stage 3 makes that
visible and handles it explicitly.
