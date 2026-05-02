# Stage 3 — TokenTracker + Tool Output Trimming

> **Context-rot prevention.** Stage 2 gave the agent memory; this stage stops
> the conversation itself from quietly poisoning the model's context window.

---

## What this stage adds

Stage 3 layers three new harness features on top of Stage 2. The ReAct loop
itself is still unchanged — every new line of code is harness-level.

```
Stage 2 loop (unchanged)
        +
┌───────────────────────────────────────────────────────────┐
│  NEW ① TokenTracker                                       │
│        Records prompt_tokens / completion_tokens per turn │
│        and cumulatively. Prints a [budget] bar before     │
│        every model call.                                  │
└───────────────────────────────────────────────────────────┘
        +
┌───────────────────────────────────────────────────────────┐
│  NEW ② Tool output trimming                               │
│        Older tool messages get their bodies replaced with │
│        a short stub. The N most recent stay full.         │
│        Per-result hard cap on entry (no single article    │
│        can ever exceed MAX_TOOL_RESULT_CHARS).            │
└───────────────────────────────────────────────────────────┘
        +
┌───────────────────────────────────────────────────────────┐
│  NEW ③ Budget guard                                       │
│        Estimate prompt size before sending. Warn at 70%   │
│        of context window, hard-stop at 90%.               │
└───────────────────────────────────────────────────────────┘
```

---

## What the model sees now (and doesn't)

The model still sees `AGENTS.md` as its operating rules and the same three
tools (`recall_memory`, `search_kb`, `get_article`). What changes is what
*lives in `messages` over time*.

Before Stage 3:

```
turn 1 → messages: [user_q]
turn 2 → messages: [user_q, asst_call, tool_result_1KB]
turn 3 → messages: [user_q, asst_call, tool_result_1KB, asst_call, tool_result_2KB]
turn 4 → messages: [..., tool_result_2KB, asst_call, tool_result_3KB]
                        ^^^^ all of this re-billed every turn
```

After Stage 3:

```
turn 4 → messages: [..., [trimmed: 1024 chars], [trimmed: 2048 chars],
                         tool_result_3KB ← still full ]
```

The shape stays the same (every `tool_call_id` still has a matching tool
message), only the body shrinks.

---

## File layout

```
support-harness/
├── articles.json           # KB data (unchanged)
├── mock_kb.py              # KB logic (unchanged)
├── stage1_bare_react.py    # Stage 1 (unchanged reference)
├── stage2_memory.py        # Stage 2 (unchanged reference)
└── stage3_tokens.py        # THIS STAGE
    └── .chroma_stage3/     # created on first run (separate from Stage 2)
```

Stage 3 uses its own ChromaDB path (`.chroma_stage3/`) so it doesn't share
state with earlier stages while you're A/B-ing them.

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
python stage3_tokens.py
```

You'll see a new banner line — `Tokens: TRACKED` — and a per-turn `[budget]`
bar like:

```
[budget] est_in≈ 1842  [██··················]   9.0% of 8192
[Turn 1] finish_reason=tool_calls  tokens=in:1801 out:34
```

After the answer:

```
[token summary]
  Per-turn token usage:
    Turn  1  in: 1801  out:  34
    Turn  2  in: 1962  out:  19
    Turn  3  in: 2390  out: 142
  Cumulative — in:6153  out:195  total:6348
```

**Session commands**

| Input     | What it does                                        |
|-----------|-----------------------------------------------------|
| question  | Runs `run_agent()`, saves result to memory          |
| `guide`   | Prints the 4-step token-tracking walkthrough        |
| `history` | Lists every question asked this session             |
| `memory`  | Shows how many cases are stored in ChromaDB         |
| `budget`  | Prints the harness budget settings                  |
| `flush`   | Wipes all stored cases, resets ChromaDB collection  |
| `quit`    | Ends the session                                    |

To wipe memory for multiple stages at once (without launching a session),
use the repo-level [`reset_memory.py`](reset_memory.py) script.

---

## Knobs

All in one block at the top of `stage3_tokens.py`:

```python
CONTEXT_WINDOW           = 8_192   # llama3.2 default; change per model
WARN_FRAC                = 0.70    # warn at 70% used
HARD_FRAC                = 0.90    # abort at 90% used
KEEP_RECENT_TOOL_OUTPUTS = 2       # how many tool results stay full
MAX_TOOL_RESULT_CHARS    = 1_500   # per-result hard cap on entry
```

---

## Suggested experiments

### 1. See the [budget] bar climb

```
You › What are the API rate limits?
You › How do I generate an API key?
You › How do I rotate my API key?
You › What does HTTP 401 mean from the API?
You › How do I authenticate to the API?
```

By the third or fourth question, the bar should creep into the warning zone
on a fresh run. (It depends on the model — bigger context windows mean it
takes longer to surface.)

### 2. Watch trimming kick in mid-run

A multi-hop question forces several tool calls in a single `run_agent()`:

```
You › I am getting 401 errors and I think my key might have expired —
      what do I do?
```

Around turn 3 or 4 you should see:

```
  [trim] replaced 1 older tool output(s) with stubs
```

That's the harness reclaiming context the model no longer needs.

### 3. Trip the hard stop on purpose

Lower the budget temporarily:

```python
HARD_FRAC = 0.20   # abort once we're past 20% — easy to trigger
```

Run any multi-hop question and watch:

```
[budget] HARD STOP — Estimated prompt size (1842 tokens) exceeds 20% ...
```

Then revert. This is the "fail loudly instead of degrading silently" payoff.

### 4. A/B against Stage 2

Run the exact same long sequence in `stage2_memory.py` and `stage3_tokens.py`
back-to-back. Stage 2 will keep climbing in tokens-per-turn; Stage 3 should
plateau because old tool outputs are stubs.

---

## What this stage does NOT do (intentionally)

| Gap | Fixed in |
|---|---|
| Trimming is by recency only — old high-value tool outputs lose first | Stage 4 (relevance-based retention) |
| No quality check on the final answer | Stage 4 |
| Token estimation is chars/4 — fine for budget, not exact | (probably never; estimation is good enough for a guard) |

---

## What's next: Stage 4

Stage 4 adds a self-critique / quality-check step before returning. The
agent can be honest about the *amount* of context now (Stage 3) and use
*relevant* context (Stage 2 memory), but a confidently-wrong answer still
ships unchallenged. Stage 4 makes the harness check the answer before the
user ever sees it.
