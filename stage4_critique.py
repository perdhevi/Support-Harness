"""
╔══════════════════════════════════════════════════════════════╗
║  Stage 4 — LLM-as-Judge Quality Gate                         ║
║  Commit: feat: self-critique pass with helpfulness/grounded  ║
║          scoring and bounded retries                         ║
╚══════════════════════════════════════════════════════════════╝

What changed from Stage 3
──────────────────────────
  1. Quality gate (LLM-as-judge).
     Before the harness returns the model's final answer to the user,
     it runs a SECOND model call in `judge` mode. The judge sees:
       • the original user query
       • a compact transcript of the tool results that were used
       • the candidate answer
     and returns a strict JSON verdict:
         {
           "helpfulness":  1-5,
           "groundedness": 1-5,
           "verdict":      "pass" | "revise",
           "reasons":      "<short critique>"
         }

  2. Bounded retry loop.
     If the verdict is "revise", the harness injects the critique
     back into `messages` as a system-style nudge and re-runs the
     ReAct loop. Up to MAX_RETRIES (default 2) attempts. If the
     final attempt still fails the gate, we return the best answer
     we have, clearly tagged with the judge's complaint.

  3. Per-attempt score banner.
     Every attempt prints its scores so you can SEE quality move
     across retries (or fail to move).

  4. Banner now shows "Quality: GATED".

Everything else — AGENTS.md injection, ChromaDB memory, token
tracking, tool-output trimming, the ReAct loop itself — is
unchanged from Stage 3.

New harness layer added
────────────────────────
  ┌────────────────────────────────────────────────────┐
  │              STAGE 4 HARNESS                       │
  │                                                    │
  │  [AGENTS.md]            ← Stage 2                  │
  │  [recall_memory]        ← Stage 2                  │
  │  [TokenTracker]         ← Stage 3                  │
  │  [trim_old_tool_outputs]← Stage 3                  │
  │  [budget guard]         ← Stage 3                  │
  │  [ReAct loop]           ← Stage 1, unchanged       │
  │       ↓                                            │
  │  [judge_answer]         ← NEW: scores draft answer │
  │       ↓ verdict=revise?                            │
  │  [retry_with_critique]  ← NEW: re-run with nudge   │
  │       ↓ pass / out of retries                      │
  │  [save_to_memory]       ← Stage 2                  │
  └────────────────────────────────────────────────────┘

Setup
──────
  pip install openai chromadb
  ollama pull llama3.2
  ollama serve  # skip if already running

Run
────
  python stage4_critique.py

Try a question whose KB hit is thin or where the model is prone to
guessing — watch the gate trip and the retry produce a tighter
answer (or fail honestly).
"""

import json
import os
import uuid
import chromadb

from mock_kb import search_kb, get_article
from model_config import MODEL, client

# ── Harness constants (Stage 3, unchanged) ────────────────────────────────────
MAX_TURNS = 12
CONTEXT_WINDOW = 8_192
WARN_FRAC = 0.70
HARD_FRAC = 0.90
KEEP_RECENT_TOOL_OUTPUTS = 2
MAX_TOOL_RESULT_CHARS = 1_500

# ── NEW IN STAGE 4 — quality-gate constants ──────────────────────────────────
# The judge scores helpfulness and groundedness on a 1-5 integer scale.
# Both scores must meet PASS_THRESHOLD for the answer to ship.
PASS_THRESHOLD = 4

# Total attempts = 1 initial + MAX_RETRIES retries. Keep this small —
# every retry is another full ReAct loop's worth of tokens.
MAX_RETRIES = 2

# How many chars of the tool transcript we hand to the judge. Keeping this
# bounded means the judge call itself does not balloon the budget.
JUDGE_TRANSCRIPT_CHARS = 2_000


# ─────────────────────────────────────────────────────────────────────────────
# AGENTS.md — loaded from agent_investigation.md (same file Stages 2–3 use).
# Stage 1 uses the simpler agent.md; diff the two to see what investigation
# rules and output schema get layered on starting at Stage 2.
# ─────────────────────────────────────────────────────────────────────────────
_AGENT_MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_investigation.md")

def _load_agent_md() -> str:
    with open(_AGENT_MD_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

AGENTS_MD = _load_agent_md()


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB episodic memory (unchanged from Stage 2, separate path)
# ─────────────────────────────────────────────────────────────────────────────
_chroma = chromadb.PersistentClient(path=".chroma_stage4")
_memory = _chroma.get_or_create_collection(
    name="support_memory",
    metadata={"hnsw:space": "cosine"},
)


def recall_memory(query: str, n_results: int = 3) -> str:
    count = _memory.count()
    if count == 0:
        return "No past cases in memory yet."

    results = _memory.query(
        query_texts=[query],
        n_results=min(n_results, count),
    )

    hits = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        original_query = meta.get("query", "unknown query")
        hits.append(f"[Case {i+1}] Query: {original_query}\nResolution:\n{doc}")

    return "\n\n---\n\n".join(hits) if hits else "No relevant past cases found."


def save_to_memory(query: str, answer: str) -> None:
    _memory.add(
        documents=[answer],
        metadatas=[{"query": query}],
        ids=[str(uuid.uuid4())],
    )


# ─────────────────────────────────────────────────────────────────────────────
# TokenTracker (unchanged from Stage 3)
# ─────────────────────────────────────────────────────────────────────────────
class TokenTracker:
    """Tiny stateful accountant for tokens consumed by the agent."""

    def __init__(self, context_window: int = CONTEXT_WINDOW):
        self.context_window = context_window
        self.turns: list[dict] = []
        self.cumulative_in = 0
        self.cumulative_out = 0

    @staticmethod
    def estimate(text: str) -> int:
        return max(1, len(text) // 4)

    def estimate_messages(self, messages: list, system: str = "") -> int:
        total = self.estimate(system)
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            total += self.estimate(str(content))
            for tc in m.get("tool_calls", []) or []:
                total += self.estimate(json.dumps(tc, default=str, ensure_ascii=False))
        return total

    def record(self, turn: int, in_tokens: int, out_tokens: int) -> None:
        self.turns.append({"turn": turn, "in": in_tokens, "out": out_tokens})
        self.cumulative_in += in_tokens
        self.cumulative_out += out_tokens

    def status(self, estimated_in: int) -> tuple[str, float]:
        frac = estimated_in / self.context_window
        if frac >= HARD_FRAC:
            return "hard", frac
        if frac >= WARN_FRAC:
            return "warn", frac
        return "ok", frac

    def format_summary(self) -> str:
        if not self.turns:
            return "  (no turns recorded)"
        lines = ["  Per-turn token usage:"]
        for t in self.turns:
            lines.append(f"    Turn {t['turn']:>2}  in:{t['in']:>5}  out:{t['out']:>4}")
        lines.append(
            f"  Cumulative — in:{self.cumulative_in}  "
            f"out:{self.cumulative_out}  "
            f"total:{self.cumulative_in + self.cumulative_out}"
        )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool output trimming (unchanged from Stage 3)
# ─────────────────────────────────────────────────────────────────────────────
TRIM_STUB = (
    "[trimmed: original was {n} chars — earlier tool result, no longer in context. "
    "Re-issue the tool call if you need this content again.]"
)


def cap_single_tool_result(result_str: str) -> str:
    if len(result_str) <= MAX_TOOL_RESULT_CHARS:
        return result_str
    head = result_str[:MAX_TOOL_RESULT_CHARS]
    return head + f"\n\n[truncated: full result was {len(result_str)} chars]"


def trim_old_tool_outputs(messages: list, keep_recent: int = KEEP_RECENT_TOOL_OUTPUTS) -> int:
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent:
        return 0

    to_trim = tool_indices[:-keep_recent]
    trimmed_count = 0
    for i in to_trim:
        body = messages[i].get("content") or ""
        if isinstance(body, str) and body.startswith("[trimmed:"):
            continue
        original_len = len(str(body))
        messages[i]["content"] = TRIM_STUB.format(n=original_len)
        trimmed_count += 1
    return trimmed_count


# ── Tool definitions (unchanged from Stage 2/3) ──────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Search episodic memory for past support resolutions similar to "
                "the current query. Call this FIRST before searching the KB."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The customer's question or issue description.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the knowledge base. Returns article summaries (id + title).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "product": {
                        "type": "string",
                        "description": "Optional product filter (e.g. 'portal', 'api', 'billing').",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_article",
            "description": "Fetch the full content of a KB article by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string"}
                },
                "required": ["article_id"],
            },
        },
    },
]


# ── Tool dispatcher (unchanged behaviour from Stage 3) ───────────────────────
def dispatch_tool(name: str, tool_input: dict) -> str:
    def _str(val):
        return val["query"] if isinstance(val, dict) and "query" in val else str(val)

    if name == "recall_memory":
        result = recall_memory(_str(tool_input.get("query", "")))

    elif name == "search_kb":
        result = search_kb(
            query=_str(tool_input.get("query", "")),
            product=tool_input.get("product"),
        )

    elif name == "get_article":
        article_id = tool_input.get("article_id", "")
        if isinstance(article_id, dict):
            article_id = article_id.get("article_id", "")
        result = get_article(str(article_id))

    else:
        result = {"error": f"Unknown tool: {name}"}

    raw = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
    return cap_single_tool_result(raw)


# ─────────────────────────────────────────────────────────────────────────────
# NEW IN STAGE 4 ① — LLM-as-judge quality gate
#
# A second model call, in a totally different role: rate the candidate
# answer on a 1-5 scale for helpfulness and groundedness, and emit a
# strict JSON verdict. The judge is deliberately given NO tools — it
# evaluates only what it sees.
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """
You are a strict quality evaluator for a customer support agent.

You will be given:
  • the customer's original question
  • a transcript of the tools the agent called and what they returned
  • the agent's final answer

You must rate the answer on TWO axes, each on an integer 1-5 scale:

  helpfulness  — does it directly address the customer's question with
                 actionable, specific guidance? (5 = perfect, 1 = useless)
  groundedness — is every factual claim supported by the tool transcript
                 (KB articles, prior cases)? Penalise hallucinations and
                 unsupported details. (5 = fully grounded, 1 = invented)

Then emit a verdict:
  • "pass"   — both scores are >= 4 AND the answer is honestly scoped
               (admitting "I don't know" when the KB is silent IS a pass).
  • "revise" — at least one score is < 4, OR the answer asserts something
               not visible in the transcript.

Respond with ONLY a JSON object, no preamble, no code fences:

{
  "helpfulness": <int 1-5>,
  "groundedness": <int 1-5>,
  "verdict": "pass" | "revise",
  "reasons": "<one short sentence — the single most important issue, or 'looks good'>"
}
""".strip()


def _build_tool_transcript(messages: list) -> str:
    """
    Compact rendering of the tool calls/results from a run, for the judge.

    We keep the structure (call → result) but cap total length so the judge
    call itself stays cheap. Older trimmed stubs are skipped — the judge
    cannot evaluate evidence that no longer exists in context.
    """
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                lines.append(f"CALL  {fn.get('name')}({fn.get('arguments', '')})")
        elif role == "tool":
            body = m.get("content") or ""
            if isinstance(body, str) and body.startswith("[trimmed:"):
                lines.append("RESULT (trimmed earlier — skipped)")
            else:
                snippet = str(body)
                if len(snippet) > 600:
                    snippet = snippet[:600] + " …"
                lines.append(f"RESULT {snippet}")
    transcript = "\n".join(lines) if lines else "(no tool calls were made)"
    if len(transcript) > JUDGE_TRANSCRIPT_CHARS:
        transcript = transcript[:JUDGE_TRANSCRIPT_CHARS] + "\n…[transcript truncated]"
    return transcript


def _coerce_score(value, default: int = 1) -> int:
    """Force the judge's score into a 1-5 int even if the model gets sloppy."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(5, n))


def judge_answer(query: str, messages: list, answer: str) -> dict:
    """
    Run the LLM-as-judge pass. Always returns a dict with the keys
    helpfulness, groundedness, verdict, reasons — even on parse failure.
    """
    transcript = _build_tool_transcript(messages)
    user_payload = (
        f"=== CUSTOMER QUESTION ===\n{query}\n\n"
        f"=== TOOL TRANSCRIPT ===\n{transcript}\n\n"
        f"=== AGENT'S ANSWER ===\n{answer}\n"
    )

    # No tools for the judge — pure evaluation.
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_payload},
        ],
        # response_format=...  # not all Ollama builds support JSON mode;
        # we parse defensively below.
    )

    raw = (response.choices[0].message.content or "").strip()

    # Be forgiving — strip ``` fences if the model adds them.
    if raw.startswith("```"):
        raw = raw.strip("`")
        # drop leading "json" if present
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        # Defensive fallback — treat unparseable judges as a soft revise.
        return {
            "helpfulness": 1,
            "groundedness": 1,
            "verdict": "revise",
            "reasons": f"judge returned invalid JSON: {raw[:120]!r}",
        }

    return {
        "helpfulness":  _coerce_score(verdict.get("helpfulness")),
        "groundedness": _coerce_score(verdict.get("groundedness")),
        "verdict":      "pass" if verdict.get("verdict") == "pass" else "revise",
        "reasons":      str(verdict.get("reasons", ""))[:300],
    }


def gate_passes(verdict: dict) -> bool:
    """Both scores must clear PASS_THRESHOLD AND verdict must say pass."""
    return (
        verdict["verdict"] == "pass"
        and verdict["helpfulness"] >= PASS_THRESHOLD
        and verdict["groundedness"] >= PASS_THRESHOLD
    )


def format_verdict(verdict: dict) -> str:
    icon = "✅" if verdict["verdict"] == "pass" else "❌"
    return (
        f"  {icon} judge: helpfulness={verdict['helpfulness']}/5  "
        f"groundedness={verdict['groundedness']}/5  "
        f"verdict={verdict['verdict']}  — {verdict['reasons']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inner ReAct loop — same as Stage 3, but factored out so the outer
# quality-gate loop can call it more than once with an evolving messages list.
# ─────────────────────────────────────────────────────────────────────────────
def _react_loop(messages: list, system_prompt: str, attempt: int) -> tuple[str, TokenTracker]:
    """
    Run one ReAct loop to completion. Mutates `messages` in place.
    Returns (final_answer, tracker). Empty string means MAX_TURNS or hard-stop.
    """
    tracker = TokenTracker()

    for turn in range(1, MAX_TURNS + 1):
        n_trimmed = trim_old_tool_outputs(messages)
        if n_trimmed:
            print(f"  [trim] replaced {n_trimmed} older tool output(s) with stubs")

        estimated_in = tracker.estimate_messages(messages, system=system_prompt)
        status, frac = tracker.status(estimated_in)
        bar = "█" * int(frac * 20) + "·" * (20 - int(frac * 20))
        print(
            f"  [budget] est_in≈{estimated_in:>5}  [{bar}] "
            f"{frac*100:5.1f}% of {tracker.context_window}"
        )

        if status == "hard":
            msg = (
                f"Estimated prompt size ({estimated_in} tokens) exceeds "
                f"{HARD_FRAC*100:.0f}% of context window ({tracker.context_window})."
            )
            print(f"  [budget] HARD STOP — {msg}")
            return "", tracker
        if status == "warn":
            print(f"  [budget] WARNING — over {WARN_FRAC*100:.0f}%, output quality may degrade")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        usage = response.usage

        in_t = getattr(usage, "prompt_tokens", 0) or 0
        out_t = getattr(usage, "completion_tokens", 0) or 0
        tracker.record(turn, in_t, out_t)
        print(f"[Attempt {attempt} · Turn {turn}] finish_reason={finish_reason}  tokens=in:{in_t} out:{out_t}")

        assistant_msg = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if finish_reason == "stop":
            return (message.content or ""), tracker

        if finish_reason == "tool_calls" and message.tool_calls:
            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                preview_in = tc.function.arguments[:80]
                print(f"  → {name}({preview_in})")

                result_str = dispatch_tool(name, tool_input)

                preview_out = result_str[:120] + ("..." if len(result_str) > 120 else "")
                print(f"  ← {preview_out}")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str,
                })
        else:
            print(f"\n[Warning] Unexpected finish_reason: {finish_reason}")
            break

    return "", tracker


# ─────────────────────────────────────────────────────────────────────────────
# NEW IN STAGE 4 ② — outer quality-gate loop
#
# This wraps the ReAct loop. After the model produces a draft answer, the
# judge scores it. If the gate fails, we inject the critique back into
# `messages` and re-run, up to MAX_RETRIES times.
# ─────────────────────────────────────────────────────────────────────────────
def run_agent(query: str) -> str:
    print(f"\n{'═' * 60}")
    print(f"  Query: {query}")
    print(f"{'═' * 60}\n")

    messages = [{"role": "user", "content": query}]
    system_prompt = AGENTS_MD

    best_answer = ""
    best_verdict: dict | None = None
    cumulative_total_tokens = 0

    for attempt in range(1, MAX_RETRIES + 2):  # 1 + MAX_RETRIES
        print(f"  ─── Attempt {attempt}/{MAX_RETRIES + 1} ───")

        answer, tracker = _react_loop(messages, system_prompt, attempt)
        cumulative_total_tokens += tracker.cumulative_in + tracker.cumulative_out

        if not answer:
            # Budget hard-stop or MAX_TURNS — no point judging an empty answer.
            print("  [gate] no answer produced this attempt; aborting retries")
            break

        # ── HARNESS: run the judge ────────────────────────────────────────────
        print("  [gate] running judge…")
        verdict = judge_answer(query, messages, answer)
        print(format_verdict(verdict))

        # Track the best one we've seen, in case all attempts fail the gate.
        if best_verdict is None or (
            verdict["helpfulness"] + verdict["groundedness"]
            > best_verdict["helpfulness"] + best_verdict["groundedness"]
        ):
            best_answer = answer
            best_verdict = verdict

        if gate_passes(verdict):
            print(f"\n[Answer · attempt {attempt} · PASSED gate]\n{answer}\n")
            print("[token summary]")
            print(tracker.format_summary())
            print(f"  Cumulative across all attempts: {cumulative_total_tokens} tokens")
            print()
            return answer

        # ── HARNESS: gate failed — inject critique and retry if budget allows ─
        if attempt > MAX_RETRIES:
            break

        critique = (
            "Your previous answer did not pass the quality gate.\n"
            f"  helpfulness:  {verdict['helpfulness']}/5\n"
            f"  groundedness: {verdict['groundedness']}/5\n"
            f"  reviewer note: {verdict['reasons']}\n\n"
            "Revise your answer. If groundedness is the issue, re-read the "
            "relevant article with `get_article` rather than restating from "
            "memory. If helpfulness is the issue, be more specific and "
            "actionable. If the KB genuinely does not cover this, say so "
            "plainly — that is a passing answer."
        )
        # The critique enters as a user message so the model treats it as a
        # new instruction within the same conversation. AGENTS.md remains
        # the immutable system prompt.
        messages.append({"role": "user", "content": critique})
        print("  [gate] revise — injecting critique and retrying\n")

    # ── HARNESS: out of retries — return our best, tagged honestly ───────────
    print("\n[gate] all attempts exhausted; returning best-scoring draft\n")
    if best_verdict is None or not best_answer:
        return "Error: no draft answer survived the loop."

    tagged = (
        f"{best_answer}\n\n"
        f"---\n"
        f"⚠ Quality gate did not pass after {MAX_RETRIES + 1} attempts.\n"
        f"   helpfulness={best_verdict['helpfulness']}/5  "
        f"groundedness={best_verdict['groundedness']}/5\n"
        f"   reviewer note: {best_verdict['reasons']}"
    )
    return tagged


# ── Interactive session ───────────────────────────────────────────────────────
def session() -> None:
    """
    REPL with the LLM-as-judge gate visible on every turn.

    Commands:
      guide             → show the 4-step quality-gate walkthrough
      quit / exit / q   → end the session
      history           → list every question asked this session
      memory            → show how many cases are stored in ChromaDB
      budget            → print the harness budget settings
      gate              → print the quality-gate settings
      flush             → wipe all stored cases, reset ChromaDB collection
      (empty line)      → skipped
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  SupportHarness — Stage 4: LLM-as-Judge Quality Gate   ║")
    print(f"║  Model  : {MODEL:<48}║")
    print("║  Memory : ChromaDB                                      ║")
    print("║  Tokens : TRACKED                                       ║")
    print("║  Quality: GATED  ← every answer scored, retried if weak║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("What's new vs Stage 3:")
    print(f"  • Every draft answer is rated 1-5 on helpfulness and groundedness.")
    print(f"  • Pass threshold: both scores ≥ {PASS_THRESHOLD}.")
    print(f"  • Failed answers get a critique back and up to {MAX_RETRIES} retries.")
    print(f"  • If all retries fail, the best draft ships with a quality warning.")
    print()
    print("Type a support question and press Enter.")
    print("Commands: 'guide' · 'history' · 'memory' · 'budget' · 'gate' · 'flush' · 'quit'\n")

    global _memory
    history: list[str] = []

    while True:
        try:
            raw = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended.")
            break

        if not raw:
            continue

        if raw.lower() in {"quit", "exit", "q"}:
            print("\nSession ended.")
            break

        if raw.lower() == "guide":
            print(f"""
┌─────────────────────────────────────────────────────────┐
│  Quality-Gate Demo — 4-step walkthrough                 │
└─────────────────────────────────────────────────────────┘

This guide shows the LLM-as-judge in action: a clean pass,
a revise+retry, and an honest failure. Run the steps in
order for the clearest contrast.

─── STEP 1 — A clean run that passes the gate ──────────

  Ask:  How do I reset my password?

  What to watch for:
  • [Attempt 1/{MAX_RETRIES + 1}] runs the normal ReAct loop.
  • [gate] judge: helpfulness=N/5  groundedness=N/5
    verdict=pass  →  answer ships immediately, no retry.

  Why: a well-supported KB hit (kb-001) plus clear actionable
  steps is exactly what the judge is looking for. Pass
  threshold: both scores ≥ {PASS_THRESHOLD}.

─── STEP 2 — A weak draft that triggers a retry ────────

  Ask:  What ports does the API use?

  What to watch for:
  • [Attempt 1] no KB article covers ports — but the model
    is very likely to fill in "443" or "HTTPS on 443" from
    its training data instead of saying so.
  • [gate] verdict=revise (groundedness fails) → critique
    injected as a user message, judge's reasons quoted back.
  • [Attempt 2] the model retries; the passing move is to
    plainly say the KB does not document this. Often it does;
    sometimes it doubles down and attempt {MAX_RETRIES + 1} kicks in.

  Why: the judge is a separate evaluation pass with no tools.
  When it spots a claim that has no support in the tool
  transcript, it fails the gate — exactly the hallucination
  pattern the harness is designed to catch.

─── STEP 3 — A genuinely unanswerable question ─────────

  Ask:  How do I export my data to BigQuery?

  What to watch for:
  • All {MAX_RETRIES + 1} attempts likely fail the gate (no KB coverage).
  • The harness returns the BEST-scoring draft, tagged with
    a "⚠ Quality gate did not pass" footer showing the
    judge's complaint.

  Why: the gate isn't a magic upgrade. When the KB is silent,
  honesty ("I don't have an article on that") is the only
  passing answer — and small models often hallucinate
  instead. Tagging makes that visible to the human reader.

─── STEP 4 — Type 'gate' to see the settings ───────────

  Prints the pass threshold ({PASS_THRESHOLD}/5), retry budget ({MAX_RETRIES}),
  and judge transcript cap. Lower PASS_THRESHOLD in
  stage4_critique.py to make the gate more permissive while
  you tune behaviour.
─────────────────────────────────────────────────────────
""")
            continue

        if raw.lower() == "history":
            if not history:
                print("  (no questions asked yet)\n")
            else:
                print("\n  Session history:")
                for i, q in enumerate(history, 1):
                    print(f"  {i}. {q}")
                print()
            continue

        if raw.lower() == "memory":
            count = _memory.count()
            print(f"\n  ChromaDB: {count} case(s) stored in .chroma_stage4/\n")
            continue

        if raw.lower() == "flush":
            count = _memory.count()
            _chroma.delete_collection("support_memory")
            _memory = _chroma.get_or_create_collection(
                name="support_memory",
                metadata={"hnsw:space": "cosine"},
            )
            print(f"\n  ChromaDB: flushed {count} case(s) — memory is now empty.\n")
            continue

        if raw.lower() == "budget":
            print(f"\n  Budget settings:")
            print(f"    Context window      : {CONTEXT_WINDOW} tokens")
            print(f"    Warn at             : {int(WARN_FRAC*100)}% ({int(CONTEXT_WINDOW*WARN_FRAC)} tokens)")
            print(f"    Hard stop at        : {int(HARD_FRAC*100)}% ({int(CONTEXT_WINDOW*HARD_FRAC)} tokens)")
            print(f"    Keep recent tools   : {KEEP_RECENT_TOOL_OUTPUTS} full, older stubbed")
            print(f"    Per-result hard cap : {MAX_TOOL_RESULT_CHARS} chars\n")
            continue

        if raw.lower() == "gate":
            print(f"\n  Quality gate settings:")
            print(f"    Pass threshold      : both scores ≥ {PASS_THRESHOLD}/5")
            print(f"    Max retries         : {MAX_RETRIES} (so up to {MAX_RETRIES + 1} attempts total)")
            print(f"    Judge transcript cap: {JUDGE_TRANSCRIPT_CHARS} chars")
            print(f"    Judge tools         : NONE (pure evaluator)\n")
            continue

        history.append(raw)
        answer = run_agent(raw)
        save_to_memory(raw, answer)
        print(f"  [harness] Answer saved to memory ({_memory.count()} total)\n")


if __name__ == "__main__":
    session()
