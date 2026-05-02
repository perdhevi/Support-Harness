"""
╔══════════════════════════════════════════════════════════════╗
║  Stage 2 — AGENTS.md Injection + ChromaDB Episodic Memory  ║
║  Commit: feat: add behavioral guide injection and memory     ║
╚══════════════════════════════════════════════════════════════╝

What changed from Stage 1
──────────────────────────
  1. AGENTS.md injected as the system prompt prefix.
     The harness controls what behavioral rules the model sees on
     every call — the model cannot modify or ignore them.

  2. ChromaDB episodic memory.
     Two new harness-level operations are added:
       a) recall_memory(query) — a tool the model can call to fetch
          similar past cases before it decides how to investigate.
       b) save_to_memory(query, answer) — called by the harness after
          every successful run, completely invisible to the model.

  3. session() banner updated to show Memory: ChromaDB.

Everything else — the ReAct loop, tool dispatch, Ollama client,
mock_kb imports — is identical to Stage 1.

New harness layer added
────────────────────────
  ┌─────────────────────────────────────────┐
  │             STAGE 2 HARNESS             │
  │                                         │
  │  [AGENTS.md]  ← injected into system    │
  │       ↓         prompt on every call    │
  │  [recall_memory] ← model can search     │
  │       ↓           past resolutions      │
  │  [ReAct loop]  ← unchanged from Stage 1 │
  │       ↓                                 │
  │  [save_to_memory] ← harness persists    │
  │                     answer after loop   │
  └─────────────────────────────────────────┘

Setup
──────
  pip install openai chromadb
  ollama pull llama3.2
  ollama serve  # skip if already running

Run
────
  python stage2_memory.py

Ask the same question twice — the second time the agent will find
the prior resolution in memory and cite it before investigating.
"""

import json
import os
import uuid
import chromadb

from mock_kb import search_kb, get_article
from model_config import MODEL, client

# ── Harness constants ─────────────────────────────────────────────────────────
MAX_TURNS = 12

# ─────────────────────────────────────────────────────────────────────────────
# NEW IN STAGE 2 ① — AGENTS.md behavioral guide
#
# Loaded from agent_investigation.md sitting next to this file. The harness
# owns this file; the model cannot modify it. Think of it as the OS loading
# a config before handing control to a process.
#
# Stage 1 reads agent.md (a simpler "support agent" prompt). Stages 2-4 read
# agent_investigation.md, a richer guide that adds: recall_memory-first rule,
# memory-citation rule, and the DIAGNOSIS / ROOT_CAUSE / RECOMMENDED_FIX
# output schema. Diff the two files to see exactly what each stage layers on.
# ─────────────────────────────────────────────────────────────────────────────
_AGENT_MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_investigation.md")

def _load_agent_md() -> str:
    with open(_AGENT_MD_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

AGENTS_MD = _load_agent_md()


# ─────────────────────────────────────────────────────────────────────────────
# NEW IN STAGE 2 ② — ChromaDB episodic memory
#
# A single persistent ChromaDB collection stores every resolved ticket.
# Each document = one (query, answer) pair.
# The model accesses it through the recall_memory tool.
# The harness saves to it automatically after each run_agent() call.
# ─────────────────────────────────────────────────────────────────────────────
_chroma = chromadb.PersistentClient(path=".chroma_stage2")
_memory = _chroma.get_or_create_collection(
    name="support_memory",
    metadata={"hnsw:space": "cosine"},
)

# Cosine-distance cutoff. ChromaDB returns 0 (identical) → 2 (opposite).
# Hits above this are dropped before the model ever sees them, so the
# relevance judgement isn't left to a small local model.
_RELEVANCE_THRESHOLD = 0.7


def recall_memory(query: str, n_results: int = 3) -> str:
    """
    Search episodic memory for past resolutions similar to query.
    Called by the model as a tool; results go into the ReAct context.
    Hits with cosine distance above _RELEVANCE_THRESHOLD are filtered out.
    """
    count = _memory.count()
    if count == 0:
        return "No past cases in memory yet."

    results = _memory.query(
        query_texts=[query],
        n_results=min(n_results, count),
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist > _RELEVANCE_THRESHOLD:
            continue
        original_query = meta.get("query", "unknown query")
        hits.append(
            f"[Case {len(hits)+1}] Query: {original_query}\n"
            f"Resolution:\n{doc}"
        )

    return "\n\n---\n\n".join(hits) if hits else "No relevant past cases found."


def save_to_memory(query: str, answer: str) -> None:
    """
    Persist a resolved ticket to ChromaDB.
    Called by the harness after run_agent() returns — never by the model.
    """
    _memory.add(
        documents=[answer],
        metadatas=[{"query": query}],
        ids=[str(uuid.uuid4())],
    )


# ── Tool definitions (adds recall_memory to Stage 1 set) ─────────────────────
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


# ── Tool dispatcher ───────────────────────────────────────────────────────────
def dispatch_tool(name: str, tool_input: dict) -> str:
    """Route a tool call to the right function and return a JSON string."""

    # Guard: local models occasionally wrap the argument in a nested dict.
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

    return json.dumps(result, ensure_ascii=False)


# ── ReAct loop ────────────────────────────────────────────────────────────────
def run_agent(query: str) -> str:
    """
    Run one support investigation.

    Differences from Stage 1:
      • system prompt now includes AGENTS_MD (harness injects it)
      • recall_memory is available as a tool
      • after returning, the caller (session) saves to memory

    The loop logic itself is unchanged.
    """
    print(f"\n{'═' * 60}")
    print(f"  Query: {query}")
    print(f"{'═' * 60}\n")

    # ── HARNESS: build initial context ────────────────────────────────────────
    # AGENTS_MD is injected here — the model receives it as its operating rules.
    messages = [
        {"role": "user", "content": query}
    ]
    system_prompt = AGENTS_MD  # harness-controlled, not user-controlled

    for turn in range(1, MAX_TURNS + 1):
        # ── HARNESS: call model ───────────────────────────────────────────────
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message   = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        usage = response.usage

        in_t  = getattr(usage, "prompt_tokens", "?")
        out_t = getattr(usage, "completion_tokens", "?")
        print(f"[Turn {turn}] finish_reason={finish_reason}  tokens=in:{in_t} out:{out_t}")

        # ── HARNESS: append assistant turn ────────────────────────────────────
        messages.append(message)

        # ── HARNESS: handle final answer ──────────────────────────────────────
        if finish_reason == "stop":
            answer = message.content or ""
            print(f"\n[Answer]\n{answer}\n")
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

                # Ollama/OpenAI format: each tool result is its own message
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str,
                })
        else:
            print(f"\n[Warning] Unexpected finish_reason: {finish_reason}")
            break

    return "Error: reached MAX_TURNS without a final answer."


# ── Interactive session ───────────────────────────────────────────────────────
def session() -> None:
    """
    Prompt-based REPL. Unlike Stage 1, the agent now has access to
    episodic memory — ask the same question twice and watch it cite
    the prior resolution on the second run.

    Commands:
      guide             → show the 4-step memory demo walkthrough
      quit / exit / q   → end the session
      history           → list every question asked this session
      memory            → show how many cases are stored in ChromaDB
      flush             → wipe all stored cases, reset ChromaDB collection
      (empty line)      → skipped
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  SupportHarness — Stage 2: Memory                       ║")
    print(f"║  Model : {MODEL:<49}║")
    print("║  Memory: ChromaDB  ← answers saved after each run      ║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("What's new vs Stage 1:")
    print("  • The agent is told to call recall_memory FIRST.")
    print("  • Every answer is saved to ChromaDB automatically.")
    print("  • Ask the same question twice — the second run cites memory.")
    print()
    print("Type a support question and press Enter.")
    print("Commands: 'guide' · 'history' · 'memory' · 'flush' · 'quit'\n")

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
            count = _memory.count()
            print("""
┌─────────────────────────────────────────────────────────┐
│  Memory Demo — 4-step walkthrough                       │
└─────────────────────────────────────────────────────────┘

This guide shows exactly what the memory layer does at each
stage. Follow the steps in order for the clearest result.
""")
            if count > 0:
                print(f"  ⚠  ChromaDB already has {count} case(s).")
                print("  Type 'flush' first if you want a clean-slate demo.\n")

            print("""\
─── STEP 1 — Cold start (memory is empty) ───────────────

  Ask:  How do I reset my password?

  What to watch for:
  • Turn 1: recall_memory → "No past cases in memory yet."
  • Turns 2-3: agent falls through to search_kb + get_article
  • [harness] line at the end confirms the answer was saved

  Why: memory is empty so the agent does a full KB search,
  exactly like Stage 1. Nothing new yet.

─── STEP 2 — Exact recall (same question again) ─────────

  Ask:  How do I reset my password?

  What to watch for:
  • Turn 1: recall_memory → returns Case 1 from ChromaDB
  • Turn 2: finish_reason=stop  ← agent skips KB entirely
  • Answer cites the prior case directly

  Why: the cosine similarity is near-perfect for an identical
  question. The agent trusts its own prior resolution and
  answers in half the turns.

─── STEP 3 — New topic (memory miss) ────────────────────

  Ask:  What are the API rate limits for the Pro plan?

  What to watch for:
  • Turn 1: recall_memory → "No relevant past cases found."
    (Cases 1 & 2 are filtered by the cosine-distance threshold
     before the model ever sees them)
  • Turns 2-3: fresh search_kb + get_article for the API topic
  • [harness] saves a new case (now 3 total)

  Why: the harness, not the model, decides what counts as a
  relevant hit. A genuinely different topic gets a full
  investigation regardless of the model's judgement.

─── STEP 4 — Semantic recall (different wording) ────────

  Ask:  I keep getting 429 errors from the API

  What to watch for:
  • Turn 1: recall_memory → returns Case 3 (rate limits)
    even though the words "rate limit" were never used
  • Turn 2: finish_reason=stop  ← answers from memory

  Why: ChromaDB uses cosine similarity on embeddings, not
  keyword matching. "429 errors" and "rate limits" land close
  in vector space, so the prior case is retrieved.

─── After all 4 steps ───────────────────────────────────

  Type 'memory' to confirm 4 cases are stored.
  Type 'flush'  to reset and repeat the demo from scratch.
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
            print(f"\n  ChromaDB: {count} case(s) stored in .chroma_stage2/\n")
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

        history.append(raw)

        # ── HARNESS: run investigation ────────────────────────────────────────
        answer = run_agent(raw)

        # ── HARNESS: persist to memory ────────────────────────────────────────
        # This is invisible to the model. It happens after the loop ends.
        save_to_memory(raw, answer)
        print(f"  [harness] Answer saved to memory ({_memory.count()} total)\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    session()
