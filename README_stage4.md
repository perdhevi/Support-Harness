# Stage 4 — LLM-as-Judge Quality Gate

> **Don't ship a confidently-wrong answer.** Stage 1–3 made the loop work,
> made it remember, and stopped it from drowning in its own context. Stage 4
> is the first stage where the harness gets an opinion on whether the
> answer is actually good enough to send.

---

## What this stage adds

Stage 4 layers two new harness features on top of Stage 3. The ReAct loop
itself is still unchanged — it's just been factored into `_react_loop()` so
the outer gate loop can call it more than once.

```
Stage 3 loop (unchanged, now inner)
        +
┌───────────────────────────────────────────────────────────┐
│  NEW ① LLM-as-judge                                       │
│        After the model says finish_reason=stop, a SECOND  │
│        call rates the answer on helpfulness (1-5) and     │
│        groundedness (1-5) and emits a strict JSON verdict.│
│        The judge has NO tools — it evaluates only what    │
│        it sees in the transcript.                         │
└───────────────────────────────────────────────────────────┘
        +
┌───────────────────────────────────────────────────────────┐
│  NEW ② Bounded retry loop                                 │
│        On "revise", the harness injects the critique back │
│        into messages as a user-side nudge and re-runs the │
│        ReAct loop. Up to MAX_RETRIES (default 2). If all  │
│        attempts fail, the best draft ships with a clearly │
│        labeled quality warning attached.                  │
└───────────────────────────────────────────────────────────┘
```

---

## How the gate works

```
run_agent("My API key stopped working")
│
├── Attempt 1
│     ├── ReAct loop produces draft A
│     └── judge(draft A) → {helpfulness: 3, groundedness: 5, verdict: "revise",
│                           reasons: "answer doesn't tell user how to rotate the key"}
│       gate fails → inject critique → attempt 2
│
├── Attempt 2
│     ├── ReAct loop produces draft B (model has read the critique)
│     └── judge(draft B) → {helpfulness: 5, groundedness: 5, verdict: "pass"}
│       gate passes → return draft B
│
└── save_to_memory(query, draft B)
```

Two important properties:

1. **The judge sees no tools.** It evaluates only the question, the tool
   transcript that the agent actually used, and the candidate answer.
   That keeps it cheap and prevents it from "fixing" a weak answer by
   doing the investigation itself.
2. **"I don't know" is a passing answer.** The judge is told explicitly
   that an honest "the KB does not cover this" is acceptable. The gate
   exists to block hallucination, not to bully the model into confidence.

---

## What the verdict looks like

The judge returns strict JSON:

```json
{
  "helpfulness":  4,
  "groundedness": 5,
  "verdict":      "pass",
  "reasons":      "Answer cites kb-010 directly and gives the rotate command."
}
```

The harness coerces scores into 1-5 ints and treats unparseable JSON as a
soft `revise` (so a flaky judge can never accidentally green-light a bad
answer).

---

## File layout

```
support-harness/
├── articles.json              # KB data (unchanged)
├── mock_kb.py                 # KB logic (unchanged)
├── stage1_bare_react.py       # Stage 1 (reference)
├── stage2_memory.py           # Stage 2 (reference)
├── stage3_tokens.py           # Stage 3 (reference)
└── stage4_critique.py         # THIS STAGE
    └── .chroma_stage4/        # created on first run
```

Stage 4 uses its own ChromaDB path (`.chroma_stage4/`) so it doesn't share
memory with earlier stages while you're A/B-ing them.

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
python stage4_critique.py
```

You'll see a new banner line — `Quality: GATED` — and per-attempt judge
scores like:

```
  ─── Attempt 1/3 ───
  [budget] est_in≈ 1842  [██··················]   9.0% of 8192
[Attempt 1 · Turn 1] finish_reason=tool_calls  tokens=in:1801 out:34
  → recall_memory({"query": "API key stopped working"})
  ← "No past cases in memory yet."
[Attempt 1 · Turn 2] finish_reason=tool_calls  tokens=in:1962 out:19
  → search_kb({"query": "API key"})
  ← [{"id":"kb-010", ...}, {"id":"kb-011", ...}]
[Attempt 1 · Turn 3] finish_reason=stop  tokens=in:2390 out:142
  [gate] running judge…
  ❌ judge: helpfulness=3/5  groundedness=5/5  verdict=revise
     — answer doesn't tell user how to actually rotate the key
  [gate] revise — injecting critique and retrying

  ─── Attempt 2/3 ───
  ...
  ✅ judge: helpfulness=5/5  groundedness=5/5  verdict=pass — looks good
```

**Session commands**

| Input     | What it does                                        |
|-----------|-----------------------------------------------------|
| question  | Runs `run_agent()`, saves the *final* answer        |
| `guide`   | Prints the 4-step quality-gate walkthrough          |
| `history` | Lists every question asked this session            |
| `memory`  | Shows how many cases are stored in ChromaDB        |
| `budget`  | Prints the harness budget settings                  |
| `gate`    | Prints the quality-gate settings                    |
| `flush`   | Wipes all stored cases, resets ChromaDB collection  |
| `quit`    | Ends the session                                    |

To wipe memory for multiple stages at once (without launching a session),
use the repo-level [`reset_memory.py`](reset_memory.py) script.

---

## Knobs

All in one block at the top of `stage4_critique.py`:

```python
PASS_THRESHOLD        = 4       # both scores must be >= this
MAX_RETRIES           = 2       # 1 initial + 2 retries = 3 attempts max
JUDGE_TRANSCRIPT_CHARS = 2_000  # cap on what the judge sees from the run
```

The Stage 3 knobs (`CONTEXT_WINDOW`, `WARN_FRAC`, `HARD_FRAC`,
`KEEP_RECENT_TOOL_OUTPUTS`, `MAX_TOOL_RESULT_CHARS`) are unchanged.

---

## Suggested experiments

### 1. Watch a draft fail and recover

Pick a question with a thin direct match in the KB:

```
You › My API key stopped working
```

On llama3.2 the first draft often skips the rotation step. The judge
catches the gap, the critique nudges the model back to `get_article`
on `kb-010`, and the second draft typically passes.

### 2. Force a guaranteed retry

Drop the threshold to make the gate hard to satisfy:

```python
PASS_THRESHOLD = 5
```

Most answers will revise at least once. Useful for watching the critique
text appear and the model adjust its tool calls.

### 3. Verify "I don't know" is a passing answer

Ask something the KB explicitly does not cover:

```
You › Do you have a dark mode?
```

A grounded "no, the KB does not cover this — escalate to product"
should pass the gate. If it doesn't, the judge prompt needs tightening.

### 4. A/B against Stage 3

Run the same question in `stage3_tokens.py` and `stage4_critique.py`
back-to-back. Stage 3 will sometimes ship an answer that mentions a
fix that isn't in the article it cited. Stage 4 should either fix it on
retry or flag it with the quality warning.

### 5. Stress the judge

Set `MAX_RETRIES = 0` and ask a borderline question. The harness will
attach the warning to the best draft and ship it tagged. This is the
"fail loudly instead of degrading silently" payoff at the answer layer
(Stage 3 was the same idea but at the budget layer).

---

## What this stage does NOT do (intentionally)

| Gap                                                              | Fixed in        |
|------------------------------------------------------------------|-----------------|
| Same model is used for agent and judge — no independence boost   | Future (open)   |
| Trimming is by recency only — old high-value evidence drops first | Future (open)   |
| No regression evals — gate decisions are anecdotal across runs   | Future (open)   |
| Judge sees a transcript, not the live KB — can't verify by lookup | Future (open)   |

Using two different models for agent vs. judge is the obvious next
upgrade — anything from a smaller judge for cost down to a stronger
judge for accuracy.

---

## What's next (Stage 5+, open)

The stage progression has now covered the four core harness concerns:

| Stage | Concern              |
|-------|----------------------|
| 1     | Loop control         |
| 2     | Memory + behaviour   |
| 3     | Context budget       |
| 4     | Output quality       |

Beyond Stage 4 the natural directions are no longer linear pain points
but optimisations and infrastructure:

- **Relevance-based tool-output retention** instead of recency-only.
- **Parallel tool dispatch** when the model emits multiple tool calls per turn.
- **Evals harness** — a fixed question set, golden answers, and pass-rate
  tracking across stages so changes to any layer can be regression-tested.
- **Multi-model judge** — run the gate against a different (smaller or
  stronger) model than the agent for independence.
- **Streaming + early termination** — stop generation when the model is
  visibly looping rather than waiting for `MAX_TURNS`.
