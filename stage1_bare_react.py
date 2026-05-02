"""
╔══════════════════════════════════════════════════════════════╗
║  Stage 1 — Bare ReAct Loop (Ollama / llama3.2)             ║
║  Commit: feat: bare ReAct loop with KB tools                ║
╚══════════════════════════════════════════════════════════════╝

Same loop logic as the Anthropic version, different wire format.
Ollama exposes an OpenAI-compatible endpoint, so we use the
openai Python client pointed at localhost.

Key format differences vs Anthropic SDK
────────────────────────────────────────
  Anthropic                         OpenAI/Ollama
  ─────────────────────────────     ──────────────────────────────
  system= kwarg on create()         {"role":"system"} in messages
  tool "input_schema" key           tool "parameters" key
  stop_reason == "tool_use"         finish_reason == "tool_calls"
  stop_reason == "end_turn"         finish_reason == "stop"
  response.content (list of blocks) response.choices[0].message
  block.type == "tool_use"          message.tool_calls (list)
  block.input (dict)                tc.function.arguments (JSON string)
  tool_result in user message       {"role":"tool"} message

Run
───
  ollama pull llama3.2
  ollama serve          # usually already running
  pip install openai
  python stage1_bare_react.py
"""

import json
import os
from mock_kb import search_kb, get_article
from model_config import MODEL, client

# ── Config ────────────────────────────────────────────────────────────────────
MAX_TURNS = 10

# ── System Prompt ─────────────────────────────────────────────────────────────
# Loaded from agent.md sitting next to this file. The harness owns this file;
# editing agent.md changes behaviour without touching code.
_AGENT_MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.md")

def _load_agent_md() -> str:
    with open(_AGENT_MD_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT = _load_agent_md()

# ── Tool Definitions ──────────────────────────────────────────────────────────
# OpenAI format: each tool is {"type": "function", "function": {...}}
# The schema key is "parameters" (not "input_schema" like Anthropic).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": (
                "Search the knowledge base for articles relevant to a support query. "
                "Returns article summaries (id + title). "
                "Follow up with get_article to read the full body of promising results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or phrase to search for.",
                    },
                    "product": {
                        "type": "string",
                        "description": (
                            "Optional product filter. "
                            "Known values: 'portal', 'api', 'billing'. "
                            "Omit to search all products."
                        ),
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
            "description": (
                "Retrieve the full content of a knowledge base article by its ID. "
                "Use this after search_kb to read the body of a promising result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {
                        "type": "string",
                        "description": "The article ID returned by search_kb (e.g. 'kb-001').",
                    },
                },
                "required": ["article_id"],
            },
        },
    },
]


# ── Tool Input Normalizer ─────────────────────────────────────────────────────
#
# Different local models produce subtly broken tool arguments. Rather than
# patching each model separately, we normalise inputs in one place so any
# model swap keeps working.
#
# Known quirks by model:
#
#   llama3.2  — full params stuffed under one key:
#               {"query": {"product": "None", "query": "reset password"}}
#               also single-key nesting: {"query": {"query": "reset password"}}
#   gemma3    — double-encoded: value is still a JSON string after outer parse
#   mistral   — missing optional keys omitted entirely (fine, .get() handles it)
#   qwen2.5   — generally clean, may include extra whitespace
#   phi3      — entire arguments payload sent as a JSON string
#
# If you hit a new pattern with a different model, add it here with a comment.

def _unwrap(value):
    """Recursively unwrap a single-key dict down to its leaf value."""
    while isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    return value


def _to_str(value) -> str:
    """Coerce a value to a clean string."""
    value = _unwrap(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                value = _unwrap(parsed)
            except json.JSONDecodeError:
                pass
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value).strip()


def _hoist_nested_params(normalized: dict) -> dict:
    """
    Detect and fix the pattern where a model stuffs all parameters as the
    value of a single key instead of at the top level.

    Example (llama3.2):
      IN:  {"query": '{"product": "None", "query": "reset password"}'}
      OUT: {"query": "reset password", "product": "None"}

    Detection rule: if a string value parses as a JSON dict AND that inner
    dict contains the same key as the outer field (e.g. outer key "query"
    appears inside the inner dict), the model almost certainly intended the
    inner dict to be the full params object — hoist it.
    """
    for k, v in normalized.items():
        if not isinstance(v, str):
            continue
        stripped = v.strip()
        if not stripped.startswith("{"):
            continue
        try:
            inner = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(inner, dict):
            continue
        # Hoist only if the outer key also appears inside the inner dict
        # (strong signal that the whole params object was nested here)
        if k in inner:
            return {
                ik: _to_str(iv) if isinstance(iv, (str, dict)) else str(iv).strip()
                for ik, iv in inner.items()
            }
    return normalized


def normalize_tool_input(raw_input) -> dict:
    """
    Normalise whatever the model sent into a clean flat dict with string values.

    Handles (in order):
      1. Entire arguments as a JSON string (phi3)
      2. Per-value unwrapping and string coercion
      3. Full params nested under a single key (llama3.2 multi-word queries)
    """
    if isinstance(raw_input, str):
        try:
            raw_input = json.loads(raw_input)
        except json.JSONDecodeError:
            return {"error": f"Could not parse tool arguments: {raw_input}"}

    if not isinstance(raw_input, dict):
        return {"error": f"Unexpected tool input type: {type(raw_input).__name__}"}

    normalized = {
        k: _to_str(v) if isinstance(v, (str, dict)) else v
        for k, v in raw_input.items()
    }

    # Pass 2: hoist if the model stuffed all params under one key
    return _hoist_nested_params(normalized)


# ── Tool Dispatcher ───────────────────────────────────────────────────────────
KNOWN_PRODUCTS = {"portal", "api", "billing"}


def dispatch_tool(name: str, raw_input: dict) -> str:
    tool_input = normalize_tool_input(raw_input)

    if "error" in tool_input:
        return json.dumps(tool_input)

    print(f"  [debug] normalized tool_input: {tool_input}")

    if name == "search_kb":
        raw_product = tool_input.get("product") or None

        # Treat the string "None" / "none" as no filter
        if isinstance(raw_product, str) and raw_product.lower() == "none":
            raw_product = None

        # Reject any product value that isn't a known product name
        if raw_product and raw_product not in KNOWN_PRODUCTS:
            print(f"  [debug] ignoring unknown product filter: {raw_product!r}")
            raw_product = None

        rows = search_kb(
            query=tool_input.get("query", ""),
            product=raw_product,
        )

        # Reformat results so the id is impossible for the model to miss.
        # Raw JSON like [{"id":"kb-007",...}] causes models to hallucinate
        # placeholder strings instead of extracting the actual id value.
        if rows and "message" not in rows[0]:
            lines = ["Found the following articles:"]
            for r in rows:
                vis = "[internal]" if r.get("is_internal") else "[public]"
                lines.append(
                    f'  id="{r["id"]}"  title="{r["title"]}"  '
                    f'product={r["product"]}  {vis}'
                )
            lines.append("")
            lines.append(
                "To read an article, call get_article with the exact id "
                'string above, e.g. get_article(article_id="kb-007").'
            )
            return "\n".join(lines)
        else:
            return "No matching articles found for that query."

    elif name == "get_article":
        result = get_article(tool_input.get("article_id", ""))
        return json.dumps(result)
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ── ReAct Loop ────────────────────────────────────────────────────────────────
def run_agent(query: str) -> str:
    """
    Run the agent on a single support query.

    Loop:
      ┌──────────────────────────────────┐
      │  POST /v1/chat/completions       │
      └────────────────┬─────────────────┘
                       │
            finish_reason == "tool_calls"?
                ┌──────┴──────┐
               YES            NO  (finish_reason == "stop")
                │              │
         dispatch tools    return answer
                │
         append {"role":"tool"} messages
                │
         └──────► loop back
    """
    print(f"\n{'═' * 60}")
    print(f"  Query: {query}")
    print(f"{'═' * 60}")

    # OpenAI format: system prompt lives in messages, not a kwarg.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]
    turn = 0

    while turn < MAX_TURNS:
        turn += 1

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason

        # Token usage (not always populated by Ollama, guard against None).
        usage = response.usage
        if usage:
            print(f"\n[Turn {turn}] finish_reason={finish_reason}  "
                  f"tokens=in:{usage.prompt_tokens} out:{usage.completion_tokens}")
        else:
            print(f"\n[Turn {turn}] finish_reason={finish_reason}")

        # ── Final answer ──────────────────────────────────────────────────────
        if finish_reason == "stop":
            answer = message.content or "(no text in response)"
            print(f"\n[Answer]\n{answer}")
            return answer

        # ── Tool calls ────────────────────────────────────────────────────────
        if finish_reason == "tool_calls":
            # 1. Append the assistant message as a plain dict so the whole
            #    messages list stays JSON-serialisable across turns.
            messages.append({
                "role": "assistant",
                "content": message.content,   # may be None or a reasoning blurb
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (message.tool_calls or [])
                ],
            })

            # 2. Dispatch each tool call and append a {"role":"tool"} message.
            #    Unlike Anthropic, each result is its own message — not a list
            #    of blocks inside a single user message.
            for tc in (message.tool_calls or []):
                name = tc.function.name
                # arguments comes back as a JSON string — parse before dispatching.
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                input_preview = tc.function.arguments[:80]
                print(f"  → {name}({input_preview})")

                result_str = dispatch_tool(name, tool_input)

                preview = result_str[:120] + ("..." if len(result_str) > 120 else "")
                print(f"  ← {preview}")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,     # must match the assistant message above
                    "content":      result_str,
                })

        else:
            print(f"\n[Warning] Unexpected finish_reason: {finish_reason}")
            break

    return "Error: reached MAX_TURNS without a final answer"


# ── Interactive Session ───────────────────────────────────────────────────────
def session():
    """
    Prompt-based REPL. Each question is a fresh run_agent() call —
    no state is shared between them.

    This is the point: watch how the agent introduces itself from scratch
    every single time, with zero awareness of what was just asked.
    That's the memory gap Stage 2 fixes.

    Commands:
      guide             → show the 3-step memory-gap walkthrough
      quit / exit / q   → end the session
      history           → print every question asked this session
      (empty line)      → skipped
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  SupportHarness — Stage 1: Bare ReAct Loop              ║")
    print(f"║  Model : {MODEL:<49}║")
    print("║  Memory: NONE  ← each question starts completely fresh  ║")
    print("╚" + "═" * 58 + "╝")
    print("\nType a support question and press Enter.")
    print("Commands: 'guide' · 'history' · 'quit'\n")

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
            print("""
┌─────────────────────────────────────────────────────────┐
│  Bare ReAct Demo — 3-step walkthrough                   │
└─────────────────────────────────────────────────────────┘

This guide shows the canonical ReAct shape — and the gap
that Stage 2 fixes. Run the steps in order.

─── STEP 1 — A normal investigation ─────────────────────

  Ask:  How do I reset my password?

  What to watch for:
  • Turn 1: search_kb(query="password reset")
            → returns the kb-001 summary
  • Turn 2: get_article(article_id="kb-001")
            → returns the full article body
  • Turn 3: finish_reason=stop  → answer to user

  Why: this is the canonical ReAct shape — search → fetch →
  answer. Three turns, every turn visible in the log.

─── STEP 2 — A follow-up the agent cannot follow ───────

  Ask:  How long is that reset link valid for?

  What to watch for:
  • The model has NO record of the previous question. It will
    either ask "what reset link?" or guess from the system
    prompt alone.
  • Nothing carries over from the previous run — same system
    prompt, empty messages list.

  Why: each run_agent() call starts from scratch. There is no
  memory layer in this stage — that is the entire point.

─── STEP 3 — Type 'history' ─────────────────────────────

  Lists every question you've asked this session. The agent
  has zero awareness of any of them. Stage 2 closes this gap
  with a ChromaDB episodic memory layer.
─────────────────────────────────────────────────────────
""")
            continue

        if raw.lower() == "history":
            if not history:
                print("  (no questions asked yet)\n")
            else:
                print("\n  Questions this session:")
                for i, q in enumerate(history, 1):
                    print(f"  {i:>2}. {q}")
                print(
                    "\n  ↑ Notice: the agent has no idea any of these were asked.\n"
                    "    Each answer was built from zero context.\n"
                    "    Stage 2 fixes this with ChromaDB episodic memory.\n"
                )
            continue

        history.append(raw)

        # Each call is completely independent — no shared state.
        # This is the entire point of Stage 1.
        run_agent(raw)

        # Nudge after the first follow-up question so the gap is obvious.
        if len(history) == 2:
            print(
                "\n  ── Memory note ─────────────────────────────────────────\n"
                "  The agent just answered two questions with no connection\n"
                "  between them. Try asking a follow-up that references your\n"
                "  first question — it will have no idea what you mean.\n"
                "  ────────────────────────────────────────────────────────\n"
            )


if __name__ == "__main__":
    session()
