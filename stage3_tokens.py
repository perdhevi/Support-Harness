"""
╔══════════════════════════════════════════════════════════════╗
║  Stage 3 — TokenTracker + Tool Output Trimming             ║
║  Commit: feat: context-rot prevention via budget + trimming ║
╚══════════════════════════════════════════════════════════════╝

What changed from Stage 2
──────────────────────────
  1. TokenTracker — a harness-level class that records, every turn,
     how many input/output tokens the model consumed and the
     cumulative usage across the whole run. Warns when the running
     conversation crosses 70% of the model's context window and
     hard-stops if it crosses 90%.

  2. Tool output trimming — long tool results (e.g. a full KB article
     body) used to live forever inside `messages` and inflate every
     subsequent turn. Stage 3 keeps only the N most recent tool
     outputs intact; older ones are replaced with a short stub like:

         [trimmed: original was 1,842 chars — reread via get_article
          if needed]

     This is invisible to the model in the sense that the *role* and
     *tool_call_id* are preserved, so the OpenAI/Ollama wire format
     stays valid; only the content shrinks.

  3. Pre-call budget estimate — before each model call we estimate the
     prompt size (chars/4 heuristic) and emit a [budget] line so you
     can SEE context growing.

  4. Banner now shows "Tokens: TRACKED".

Everything else — AGENTS.md injection, ChromaDB memory, the ReAct
loop, the tool dispatcher — is unchanged from Stage 2.

New harness layer added
────────────────────────
  ┌─────────────────────────────────────────────┐
  │              STAGE 3 HARNESS                │
  │                                             │
  │  [AGENTS.md]      ← Stage 2                 │
  │  [recall_memory]  ← Stage 2                 │
  │  [ReAct loop]     ← Stage 1                 │
  │       ↓                                     │
  │  [TokenTracker]   ← NEW: per-turn + total   │
  │  [trim_old_tool_outputs] ← NEW: keep recent │
  │  [budget guard]   ← NEW: warn / hard-stop   │
  │       ↓                                     │
  │  [save_to_memory] ← Stage 2                 │
  └─────────────────────────────────────────────┘

Setup
──────
  pip install openai chromadb
  ollama pull llama3.2
  ollama serve  # skip if already running

Run
────
  python stage3_tokens.py

Try a long-running session — keep asking questions and watch the
[budget] line. Compare with Stage 2 where context grew silently.
"""

import json
import os
import uuid
import chromadb

from mock_kb import search_kb, get_article
from model_config import MODEL, client

# ── Harness constants ─────────────────────────────────────────────────────────
MAX_TURNS = 12

# llama3.2 default context window. Override here if you swap models.
CONTEXT_WINDOW = 8_192

# Soft and hard budget thresholds (fraction of CONTEXT_WINDOW).
WARN_FRAC = 0.70
HARD_FRAC = 0.90

# Tool-output trimming: keep this many of the most recent tool messages
# at full size. Older tool messages get their content replaced with a stub.
KEEP_RECENT_TOOL_OUTPUTS = 2

# Per-tool-result cap. If a single result is larger than this we trim it
# down to the cap before it ever enters the conversation.
MAX_TOOL_RESULT_CHARS = 1_500


# ─────────────────────────────────────────────────────────────────────────────
# AGENTS.md — loaded from agent_investigation.md (same file Stage 2 uses).
# Stage 1 uses the simpler agent.md; diff the two to see what investigation
# rules and output schema get layered on starting at Stage 2.
# ─────────────────────────────────────────────────────────────────────────────
_AGENT_MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_investigation.md")

def _load_agent_md() -> str:
    with open(_AGENT_MD_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

AGENTS_MD = _load_agent_md()


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB episodic memory (unchanged from Stage 2)
# ─────────────────────────────────────────────────────────────────────────────
_chroma = chromadb.PersistentClient(path=".chroma_stage3")
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
# NEW IN STAGE 3 ① — TokenTracker
#
# Tracks token usage per turn and cumulatively across one run_agent() call.
# Estimates use chars/4 as a conservative proxy when actual usage isn't
# available from the API yet (i.e. before the call goes out).
# ─────────────────────────────────────────────────────────────────────────────
class TokenTracker:
    """Tiny stateful accountant for tokens consumed by the agent."""

    def __init__(self, context_window: int = CONTEXT_WINDOW):
        self.context_window = context_window
        self.turns: list[dict] = []  # [{turn, in, out}]
        self.cumulative_in = 0
        self.cumulative_out = 0

    @staticmethod
    def estimate(text: str) -> int:
        """Conservative chars/4 estimate. Off by maybe 20%, fine for budget warnings."""
        return max(1, len(text) // 4)

    def estimate_messages(self, messages: list, system: str = "") -> int:
        """Estimate the prompt size for one model call."""
        total = self.estimate(system)
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, list):
                # Handle structured content blocks (rare with Ollama, defensive).
                content = json.dumps(content, ensure_ascii=False)
            total += self.estimate(str(content))
            # Tool calls live on assistant messages and add tokens too.
            for tc in m.get("tool_calls", []) or []:
                total += self.estimate(json.dumps(tc, default=str, ensure_ascii=False))
        return total

    def record(self, turn: int, in_tokens: int, out_tokens: int) -> None:
        self.turns.append({"turn": turn, "in": in_tokens, "out": out_tokens})
        self.cumulative_in += in_tokens
        self.cumulative_out += out_tokens

    def status(self, estimated_in: int) -> tuple[str, float]:
        """Return ('ok'|'warn'|'hard', frac_used) for a given estimated input size."""
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
# NEW IN STAGE 3 ② — Tool output trimming
#
# Walk the messages list and replace older tool outputs with a stub. We keep
# the message slot and tool_call_id so the wire format stays valid (the model
# still sees that a tool was called and "responded"); we just shrink the body.
# ─────────────────────────────────────────────────────────────────────────────
TRIM_STUB = (
    "[trimmed: original was {n} chars — earlier tool result, no longer in context. "
    "Re-issue the tool call if you need this content again.]"
)


def cap_single_tool_result(result_str: str) -> str:
    """Hard-cap a single tool result before it enters the conversation."""
    if len(result_str) <= MAX_TOOL_RESULT_CHARS:
        return result_str
    head = result_str[:MAX_TOOL_RESULT_CHARS]
    return head + f"\n\n[truncated: full result was {len(result_str)} chars]"


def trim_old_tool_outputs(messages: list, keep_recent: int = KEEP_RECENT_TOOL_OUTPUTS) -> int:
    """
    Replace the content of older tool messages with a short stub. Mutates
    `messages` in place. Returns the number of messages actually trimmed.
    """
    # Indices of every {"role":"tool"} message, in order.
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent:
        return 0

    to_trim = tool_indices[:-keep_recent]  # everything except the last `keep_recent`
    trimmed_count = 0
    for i in to_trim:
        body = messages[i].get("content") or ""
        # Already trimmed? Skip — content starts with the stub prefix.
        if isinstance(body, str) and body.startswith("[trimmed:"):
            continue
        original_len = len(str(body))
        messages[i]["content"] = TRIM_STUB.format(n=original_len)
        trimmed_count += 1
    return trimmed_count


# ── Tool definitions (unchanged from Stage 2) ────────────────────────────────
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


# ── Tool dispatcher (unchanged behaviour, but caps individual results) ───────
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
    # NEW IN STAGE 3: per-result hard cap so a single big article can't blow
    # the budget before trimming even gets a chance to run.
    return cap_single_tool_result(raw)


# ── ReAct loop ────────────────────────────────────────────────────────────────
def run_agent(query: str) -> str:
    """
    Run one support investigation with token tracking and tool-output trimming.

    Differences from Stage 2:
      • A TokenTracker accumulates per-turn usage and prints a [budget] line
        before every model call.
      • trim_old_tool_outputs() runs every turn after the first 2 tool calls,
        replacing older tool message bodies with a stub.
      • If the estimated prompt crosses HARD_FRAC of the context window we
        bail out cleanly with a helpful error rather than silently degrading.
    """
    print(f"\n{'═' * 60}")
    print(f"  Query: {query}")
    print(f"{'═' * 60}\n")

    messages = [{"role": "user", "content": query}]
    system_prompt = AGENTS_MD
    tracker = TokenTracker()

    for turn in range(1, MAX_TURNS + 1):
        # ── HARNESS: trim older tool outputs before sending ──────────────────
        n_trimmed = trim_old_tool_outputs(messages)
        if n_trimmed:
            print(f"  [trim] replaced {n_trimmed} older tool output(s) with stubs")

        # ── HARNESS: pre-call budget estimate ─────────────────────────────────
        estimated_in = tracker.estimate_messages(messages, system=system_prompt)
        status, frac = tracker.status(estimated_in)
        bar = "█" * int(frac * 20) + "·" * (20 - int(frac * 20))
        print(f"  [budget] est_in≈{estimated_in:>5}  [{bar}] {frac*100:5.1f}% of {tracker.context_window}")

        if status == "hard":
            msg = (
                f"Estimated prompt size ({estimated_in} tokens) exceeds "
                f"{HARD_FRAC*100:.0f}% of context window ({tracker.context_window}). "
                "Aborting before quality degrades."
            )
            print(f"  [budget] HARD STOP — {msg}")
            return f"Error (budget exceeded): {msg}"
        if status == "warn":
            print(f"  [budget] WARNING — over {WARN_FRAC*100:.0f}%, output quality may degrade")

        # ── HARNESS: call model ───────────────────────────────────────────────
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
        print(f"[Turn {turn}] finish_reason={finish_reason}  tokens=in:{in_t} out:{out_t}")

        # ── HARNESS: append assistant turn ────────────────────────────────────
        # Convert to a plain dict so the messages list stays JSON-serialisable
        # across turns and our trimming pass can mutate it freely.
        assistant_msg = {
            "role": "assistant",
            "content": message.content,
        }
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

        # ── HARNESS: handle final answer ──────────────────────────────────────
        if finish_reason == "stop":
            answer = message.content or ""
            print(f"\n[Answer]\n{answer}\n")
            print("[token summary]")
            print(tracker.format_summary())
            print()
            return answer

        # ── HARNESS: dispatch tool calls ──────────────────────────────────────
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

    print("[token summary]")
    print(tracker.format_summary())
    return "Error: reached MAX_TURNS without a final answer."


# ── Interactive session ───────────────────────────────────────────────────────
def session() -> None:
    """
    Prompt-based REPL with token tracking visible on every turn.

    Commands:
      guide             → show the 4-step token-tracking walkthrough
      quit / exit / q   → end the session
      history           → list every question asked this session
      memory            → show how many cases are stored in ChromaDB
      budget            → print the harness budget settings
      flush             → wipe all stored cases, reset ChromaDB collection
      (empty line)      → skipped
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  SupportHarness — Stage 3: TokenTracker + Trimming      ║")
    print(f"║  Model : {MODEL:<49}║")
    print("║  Memory: ChromaDB                                       ║")
    print("║  Tokens: TRACKED  ← per-turn + cumulative + budget bar  ║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("What's new vs Stage 2:")
    print(f"  • [budget] line before every model call (warn at {int(WARN_FRAC*100)}%, hard-stop at {int(HARD_FRAC*100)}%).")
    print(f"  • Older tool outputs trimmed automatically — only the {KEEP_RECENT_TOOL_OUTPUTS} most recent stay full.")
    print(f"  • Each tool result is hard-capped at {MAX_TOOL_RESULT_CHARS} chars on entry.")
    print()
    print("Type a support question and press Enter.")
    print("Commands: 'guide' · 'history' · 'memory' · 'budget' · 'flush' · 'quit'\n")

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
│  Token-Tracking Demo — 4-step walkthrough               │
└─────────────────────────────────────────────────────────┘

This guide makes context growth visible: the [budget] bar
before every model call, the per-turn token summary, and the
[trim] line that fires once older tool outputs get stubbed.

─── STEP 1 — A normal run, with a budget bar ────────────

  Ask:  How do I configure SSO?

  What to watch for:
  • Each turn prints a [budget] line BEFORE the model call
    showing estimated input tokens and a 20-char █/· bar.
  • At the end, the per-turn summary lists in/out tokens and
    a cumulative total for the whole run.

  Why: Stage 2 grew context silently. Stage 3 makes every
  byte visible so you can SEE when the budget starts to
  pinch instead of finding out via degraded answers.

─── STEP 2 — Watch context grow across runs ─────────────

  Ask:  What are the API rate limits for the Pro plan?
  Ask:  How do I set up 2FA?

  What to watch for:
  • Within a single run, the [budget] bar lengthens each turn
    as more tool results enter the conversation.
  • Between runs the messages list resets, so the bar starts
    short again — but ChromaDB memory grows (type 'memory').

  Why: each tool call costs tokens, and earlier tool outputs
  stay in context until the trimmer kicks in.

─── STEP 3 — Trigger the trimmer ────────────────────────

  Ask a question that needs 3+ get_article calls, e.g.:
        Compare SSO and 2FA — which should I enable?

  What to watch for:
  • After the {KEEP_RECENT_TOOL_OUTPUTS + 1}rd tool result you should see a line:
        [trim] replaced N older tool output(s) with stubs
  • The stubbed messages still occupy a slot but their bodies
    shrink to a one-line "[trimmed: original was X chars …]".
  • The [budget] bar stops growing as fast.

  Why: keeping every tool body forever inflates every later
  turn. Stage 3 keeps only the {KEEP_RECENT_TOOL_OUTPUTS} most recent intact.

─── STEP 4 — Type 'budget' to see the settings ──────────

  Prints the warn ({int(WARN_FRAC*100)}%) and hard-stop ({int(HARD_FRAC*100)}%) thresholds plus the
  trim configuration. Tweak the constants at the top of
  stage3_tokens.py if you want to see warnings or aborts kick
  in sooner.
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
            print(f"\n  ChromaDB: {count} case(s) stored in .chroma_stage3/\n")
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

        history.append(raw)
        answer = run_agent(raw)
        save_to_memory(raw, answer)
        print(f"  [harness] Answer saved to memory ({_memory.count()} total)\n")


if __name__ == "__main__":
    session()
