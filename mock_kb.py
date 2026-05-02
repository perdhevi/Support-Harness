"""
mock_kb.py — Knowledge base search logic for all SupportHarness stages.

Data lives in articles.json (edit that file to add/change articles).
This file is pure logic: load → index → search → retrieve.

Every stage imports:
    from mock_kb import search_kb, get_article
"""

import json
import re
from pathlib import Path

# ── Load articles from JSON ───────────────────────────────────────────────────
# Keyed by article ID for O(1) get_article() lookups.
_DATA_FILE = Path(__file__).parent / "articles.json"

def _load() -> dict[str, dict]:
    with open(_DATA_FILE, encoding="utf-8") as f:
        rows = json.load(f)
    return {row["id"]: row for row in rows}

ARTICLES: dict[str, dict] = _load()

# Derive the valid product set from the data so it stays in sync automatically.
KNOWN_PRODUCTS: set[str] = {a["product"] for a in ARTICLES.values()}


# ── Search helpers ────────────────────────────────────────────────────────────

# Load synonym groups from synonyms.json.
# Each group is a list of equivalent words — edit that file to add new variants.
_SYNONYMS_FILE = Path(__file__).parent / "synonyms.json"

def _load_synonym_map() -> dict[str, frozenset[str]]:
    with open(_SYNONYMS_FILE, encoding="utf-8") as f:
        groups = json.load(f)
    mapping: dict[str, frozenset[str]] = {}
    for group in groups:
        fs = frozenset(group)
        for word in group:
            mapping[word] = fs
    return mapping

_SYNONYM_MAP: dict[str, frozenset[str]] = _load_synonym_map()


def _tokenize(text: str) -> list[str]:
    """
    Split text into lowercase alphanumeric tokens, stripping punctuation.
    'password-reset' and 'password reset' both produce ['password', 'reset'].
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _tokenize_query(text: str) -> list[frozenset[str]]:
    """
    Tokenize a search query and expand each token into its synonym group.
    Returns a list of frozensets — each position holds all acceptable matches.
    Only applied to queries, never to article text.

    Example:
      "enforce 2FA" → [frozenset({"enforce","enforcing","enforced",...}), frozenset({"2fa"})]
    """
    result = []
    for t in _tokenize(text):
        result.append(_SYNONYM_MAP.get(t, frozenset({t})))
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def search_kb(
    query: str,
    product: str | None = None,
    include_internal: bool = False,
) -> list[dict]:
    """
    Token-based search across all articles.

    Each query token must appear in the article (AND logic), so token order
    doesn't matter — 'password reset' and 'reset password' return the same
    results. Results are ranked: title matches score 2×, content matches 1×.

    Returns article summaries only (id, title, product, is_internal).
    Use get_article() to fetch the full body of a result.

    Args:
        query:            Search string; each word matched independently.
        product:          Optional filter — must be a value from KNOWN_PRODUCTS.
        include_internal: Include AGENTS-only articles when True.
    """
    query_tokens = _tokenize_query(query)
    # Filter out any positions that somehow ended up empty
    query_tokens = [qt for qt in query_tokens if qt]
    if not query_tokens:
        return [{"message": "No matching articles found"}]

    scored = []

    for article in ARTICLES.values():
        if product and article["product"] != product:
            continue
        if article["is_internal"] and not include_internal:
            continue

        title_tokens   = _tokenize(article["title"])
        content_tokens = _tokenize(article["content"])
        all_tokens     = set(title_tokens + content_tokens)

        # ALL query tokens must be present (AND — order-independent).
        # Each query position is a frozenset of synonyms — any one matching is enough.
        if not all(any(v in all_tokens for v in qt) for qt in query_tokens):
            continue

        # Score: title hits 2×, content hits 1×
        # Use the first synonym variant that actually appears for scoring.
        score = sum(
            2 if any(v in title_tokens for v in qt)
            else 1
            for qt in query_tokens
        )

        scored.append((score, {
            "id":          article["id"],
            "title":       article["title"],
            "product":     article["product"],
            "is_internal": article["is_internal"],
        }))

    if not scored:
        return [{"message": "No matching articles found"}]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]


def get_article(article_id: str) -> dict:
    """
    Retrieve the full content of an article by ID.
    Returns an error dict if the ID does not exist.
    """
    article = ARTICLES.get(article_id)
    if not article:
        return {"error": f"Article '{article_id}' not found"}
    return article


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loaded {len(ARTICLES)} articles across products: {sorted(KNOWN_PRODUCTS)}\n")

    print("=== search: 'password reset' ===")
    for r in search_kb("password reset"):
        vis = "🔒" if r["is_internal"] else "🌐"
        print(f"  {vis} [{r['id']}] {r['title']} ({r['product']})")

    print("\n=== search: 'reset password' (should match same) ===")
    for r in search_kb("reset password"):
        vis = "🔒" if r["is_internal"] else "🌐"
        print(f"  {vis} [{r['id']}] {r['title']} ({r['product']})")

    print("\n=== search: 'payment' ===")
    for r in search_kb("payment"):
        print(f"  [{r['id']}] {r['title']}")

    print("\n=== search: 'api key' (product=api) ===")
    for r in search_kb("api key", product="api"):
        print(f"  [{r['id']}] {r['title']}")

    print("\n=== search with include_internal=True: 'login' ===")
    for r in search_kb("login", include_internal=True):
        vis = "🔒" if r["is_internal"] else "🌐"
        print(f"  {vis} [{r['id']}] {r['title']}")

    print("\n=== get_article: kb-011 ===")
    article = get_article("kb-011")
    print(f"  {article['title']}\n  {article['content'][:120]}...")
