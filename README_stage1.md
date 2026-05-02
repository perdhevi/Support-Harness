# SupportHarness — Stage 1: Bare ReAct Loop

> **The simplest possible agent harness.** Model + tools + loop. Nothing else.
> The goal is to see the raw mechanics before adding any infrastructure on top.

---

## What this stage is

Stage 1 implements a single function: `run_agent(query)`. It sends a support
query to a local LLM, lets the model call KB search tools, and returns a final
answer. There is no memory, no token tracking, no quality checking — those
come in later stages.

The loop is driven entirely by `finish_reason`:

```
user query
    │
    ▼
┌───────────────────────────┐
│  POST /v1/chat/completions │  ← Ollama (llama3.2)
└────────────┬──────────────┘
             │
    finish_reason == "tool_calls"?
       ┌─────┴─────┐
      YES          NO  ──► return answer
       │
  dispatch tools
  append {"role":"tool"} messages
       │
       └──────────► loop back
```

Two tools are available to the model:

| Tool | What it does |
|---|---|
| `search_kb(query, product?)` | Returns matching article summaries (id + title only) |
| `get_article(article_id)` | Returns the full body of one article |

The two-step design (`search` then `get`) is intentional — it forces the model
to make an explicit retrieval decision and produces a visible, auditable trace.

---

## File layout

```
support-harness/
├── articles.json           # KB data — edit this to add or change articles
├── mock_kb.py              # KB logic — loads articles.json, exposes search_kb() and get_article()
└── stage1_bare_react.py    # the bare ReAct loop (this stage)
```

`articles.json` and `mock_kb.py` are intentionally separate. The JSON is pure
data that anyone can edit without touching Python. The Python is pure logic
that never hardcodes article content.

---

## Setup

**1. Pull the model**
```bash
ollama pull llama3.2
ollama serve      # skip if already running
```

**2. Install the Python dependency**
```bash
pip install openai
```

That's it — no API key needed, everything runs locally.

---

## Run it

```bash
python stage1_bare_react.py
```

This starts an interactive session — type any question and press Enter.

```
╔══════════════════════════════════════════════════════════╗
║  SupportHarness — Stage 1: Bare ReAct Loop              ║
║  Model : llama3.2 via Ollama                            ║
║  Memory: NONE  ← each question starts completely fresh  ║
╚══════════════════════════════════════════════════════════╝

Type a support question and press Enter.
Commands: 'guide' · 'history' · 'quit'

You › How do I reset my password?

════════════════════════════════════════════════════════════
  Query: How do I reset my password?
════════════════════════════════════════════════════════════

[Turn 1] finish_reason=tool_calls  tokens=in:312 out:28
  [debug] normalized tool_input: {'query': 'reset password'}
  → search_kb({"query": "reset password"})
  ← [{"id": "kb-001", "title": "How to reset your password", ...}]

[Turn 2] finish_reason=tool_calls  tokens=in:410 out:22
  [debug] normalized tool_input: {'article_id': 'kb-001'}
  → get_article({"article_id": "kb-001"})
  ← {"id": "kb-001", "title": "How to reset your password", "content": "..."}

[Turn 3] finish_reason=stop  tokens=in:601 out:89

[Answer]
To reset your password, click 'Forgot Password' on the login page...

You › _
```

**Session commands**

| Input | What it does |
|---|---|
| Any question | Runs `run_agent()` fresh from zero context |
| `guide` | Prints a 3-step walkthrough that demonstrates the memory gap |
| `history` | Lists every question asked this session, with a reminder that the agent remembers none of them |
| `quit` / `exit` / `q` | Ends the session |

---

## What to observe

**The token count climbing each turn.** Every turn reprints the full
conversation history. By turn 3 the input token count is roughly 2× turn 1.
This is context bloat in action — Stage 3 addresses it.

**The two-step search → fetch pattern.** The model usually calls `search_kb`
first to get a list of IDs, then calls `get_article` on the most relevant one.
Sometimes it skips the fetch and answers from the summary alone — that's a
quality gap Stage 4 addresses.

**`finish_reason` as the loop controller.** There is no other exit condition
except `MAX_TURNS = 10`. The model decides when it has enough information.

**The `[debug]` line on every tool call.** Shows exactly what the model sent
after normalization. Useful for spotting model-specific argument quirks.

---

## Knowledge base contents

19 articles across 5 products. 3 are internal (agents-only) and hidden from
search by default.

| ID | Title | Product | Visible |
|---|---|---|---|
| kb-001 | How to reset your password | portal | 🌐 public |
| kb-002 | API rate limits — Pro and Enterprise plans | api | 🌐 public |
| kb-003 | Billing cycle and invoice generation | billing | 🌐 public |
| kb-004 | SSO configuration guide (SAML 2.0) | portal | 🌐 public |
| kb-005 | [INTERNAL] Escalation runbook — repeated login failures | portal | 🔒 internal |
| kb-006 | Webhook delivery failures and retry policy | api | 🌐 public |
| kb-007 | Setting up two-factor authentication (2FA) | portal | 🌐 public |
| kb-008 | Inviting team members and managing roles | portal | 🌐 public |
| kb-009 | [INTERNAL] Account deletion and data purge process | portal | 🔒 internal |
| kb-010 | API authentication — generating and rotating API keys | api | 🌐 public |
| kb-011 | Common API error codes and how to resolve them | api | 🌐 public |
| kb-012 | [INTERNAL] API abuse detection and key suspension | api | 🔒 internal |
| kb-013 | How to update your payment method | billing | 🌐 public |
| kb-014 | Upgrading or downgrading your subscription plan | billing | 🌐 public |
| kb-015 | Refund policy and how to request a refund | billing | 🌐 public |
| kb-016 | Getting started with the mobile app | mobile | 🌐 public |
| kb-017 | Managing push notification preferences | mobile | 🌐 public |
| kb-018 | Connecting your Slack workspace | integrations | 🌐 public |
| kb-019 | Using Zapier to automate workflows | integrations | 🌐 public |

---

## Suggested questions

Questions are grouped by product, then by scenario type. Each maps to one or
more specific articles in `articles.json`.

---

### 🔐 Portal — account & access

**Direct match** — should resolve in 2–3 turns with a grounded answer.

```
How do I reset my password?
I forgot my password, how do I get back in?
How do I set up two-factor authentication?
How do I enable 2FA on my account?
How do I invite someone to my team?
What roles are available for team members?
How do I configure SSO with Okta?
How do I set up SAML single sign-on?
```

**Edge cases** — ambiguous wording, partial context, likely to surface gaps.

```
I can't log in — what should I do?
My password reset email never arrived
The SSO login is not working for my team
I locked myself out of my account
How do I remove a team member?
Can a Viewer create new items?
```

---

### 🔌 API — developer

**Direct match**

```
What are the API rate limits?
How do I generate an API key?
How do I rotate my API key?
My webhooks are failing — what is the retry policy?
I am getting a 429 error from the API
What does HTTP 401 mean from the API?
What does a 422 error mean?
```

**Edge cases**

```
How do I authenticate to the API?
My API key stopped working
How long does the system retry a failed webhook?
I am getting 500 errors — is that my fault or yours?
How do I replay a failed webhook delivery?
What headers does the API return for rate limits?
```

---

### 💳 Billing — payments & plans

**Direct match**

```
When is my invoice generated?
How do I update my credit card?
How do I upgrade my plan?
How do I downgrade my subscription?
Can I get a refund?
Where can I download my past invoices?
```

**Edge cases**

```
My payment failed — what happens next?
I was charged but did not use the service this month
How long does a refund take to process?
What happens to my data if I cancel?
I want to switch from monthly to annual billing
My card was declined — how do I retry the payment?
```

---

### 📱 Mobile — app

**Direct match**

```
How do I get started with the mobile app?
Which devices does the mobile app support?
How do I turn off push notifications?
I am not receiving any push notifications
```

**Edge cases**

```
Can I manage billing from the mobile app?
How do I sign in with SSO on mobile?
The app is not sending me notifications even though they are enabled
```

---

### 🔗 Integrations — Slack & Zapier

**Direct match**

```
How do I connect Slack to my account?
What Slack slash commands are available?
How do I set up a Zapier workflow?
What triggers are available in Zapier?
```

**Edge cases**

```
My Slack notifications stopped after I reconnected my workspace
Can I use Zapier on the Free plan?
How do I disconnect the Slack integration?
What actions can Zapier perform on items?
```

---

### 🔀 Multi-hop — requires combining two articles

These questions span more than one article. The model needs to search and
fetch more than once to build a complete answer. Watch whether it makes the
second tool call or stops too early.

```
I can't log in and I think it might be an SSO problem — where do I start?
What is the difference in rate limits between Pro and Enterprise, and what should I do when I hit the limit?
How do I set up 2FA and also enforce it for my whole team?
I want to add a new team member — what role should I give them if they only need to read data?
My webhook failed and I want to replay it — what is the process and what should I check first?
How do I connect Slack and also get notified about billing events?
I am getting 401 errors and I think my key might have expired — what do I do?
Can I downgrade my plan and still use the Zapier integration?
```

---

### ❓ No match — outside the KB

Watch what happens: does the model hallucinate an answer, admit it doesn't
know, or keep looping until MAX_TURNS?

```
How do I export all my data to CSV?
What is the uptime SLA for Enterprise customers?
Do you have a dark mode?
How do I set up a custom domain?
Can I white-label the portal?
Is there an audit log for admin actions?
What programming languages does the SDK support?
How do I set up IP allowlisting?
Is there a GraphQL API?
How do I embed the portal in my own website?
```

---

### 🔒 Internal articles — visibility filter

Each of these has a matching article in the KB but it is `is_internal: true`,
so it is hidden from search by default. With default settings the model should
find no useful result and say so honestly.

```
A customer has had 5 failed login attempts — what do I do?
How do I unlock a soft-locked account?
How do I delete a customer account and purge their data?
A customer wants their data deleted under GDPR — what is the process?
An API key is being used abusively — what are the thresholds for suspension?
```

**To unlock internal articles:** open `mock_kb.py` and change the
`include_internal` default to `True`, then run the same questions again.
Articles kb-005 (login escalation), kb-009 (account deletion), and
kb-012 (API abuse) should now appear in search results.

---

## Things to tweak

**Swap the model**

```python
# in stage1_bare_react.py
MODEL = "gemma3"        # different tool-call serialization quirks
MODEL = "mistral"       # tends to omit optional parameters
MODEL = "qwen2.5"       # strong structured output, clean tool calls
MODEL = "llama3.1"      # larger, slower, generally more reliable
```

**Add a `list_products()` tool**

Lets the model discover what products exist before filtering. Add it to
`TOOLS` and `dispatch_tool`, importing `KNOWN_PRODUCTS` from `mock_kb`.

```
What billing articles do you have?
Show me everything related to integrations
```

**Lower MAX_TURNS to 2**

```python
MAX_TURNS = 2
```

Forces the model to answer after at most one tool call. Does it search or
fetch directly? Does answer quality drop noticeably on multi-hop questions?

**Add a new article**

Open `articles.json`, append a new object, and immediately test it:

```json
{
  "id": "kb-020",
  "title": "How to enable audit logging",
  "product": "portal",
  "is_internal": false,
  "content": "Audit logs record all admin actions..."
}
```

No code changes needed — `mock_kb.py` loads the file at startup.

**Print the full messages list**

Add this inside the loop to see exactly what the model receives each turn:

```python
import json
print(json.dumps(messages, indent=2, default=str))
```

Watch how fast context grows. This is the clearest possible motivation for
the token budget work in Stage 3.

---

## What this stage does NOT do (intentionally)

| Gap | Fixed in |
|---|---|
| No memory — agent forgets every run | Stage 2 |
| No behavioural guidelines — system prompt is hardcoded | Stage 2 (AGENTS.md) |
| No token awareness — context grows silently | Stage 3 |
| No quality check — weak answers ship as-is | Stage 4 |

---

## The memory gap — experiments to run

These sequences are specifically designed to expose what the agent *can't* do
at this stage. Run them in the interactive session and observe.

### Sequence 1 — Direct follow-up

```
You › How do I reset my password?
  ... agent answers ...

You › What if I don't get the email?
```

The second question is a natural follow-up. A human support agent would
immediately connect it to the password reset context. This agent has no idea
what "the email" refers to — it will either search blindly or ask for
clarification as if the first question never happened.

---

### Sequence 2 — Reference back to a previous answer

```
You › What are the API rate limits for the Pro plan?
  ... agent answers with 1,000 req/min ...

You › What about the Enterprise plan?
  ... agent answers ...

You › Summarise what you just told me about both plans
```

The last prompt asks the agent to refer back to its own prior answers.
It cannot — each `run_agent()` call starts with a blank messages list.
It will either hallucinate a summary or search the KB again from scratch.

---

### Sequence 3 — Pronoun reference

```
You › How do I set up SSO with Okta?
  ... agent answers ...

You › Can I do the same thing with Azure AD?
```

"The same thing" references the SSO setup process from the prior turn.
The agent receives only the second question with no context for what
"the same thing" means.

---

### Sequence 4 — Correction loop

```
You › How do I configure SSO?
  ... agent answers ...

You › You missed the part about testing — what was the testing step?
```

The agent cannot refer back to what it said, so it cannot identify what
was "missed". It will re-answer the whole question or be confused by
"you missed".

---

### Sequence 5 — Cross-topic reference

```
You › How do I invite a new team member?
  ... agent answers ...

You › What plan do I need to give them API access?
  ... agent answers about plans ...

You › OK so based on what you told me, should I upgrade first or invite them first?
```

The final question requires reasoning over both prior answers together.
The agent has neither — it will treat it as a completely new question.

---

### After running these — type `history`

```
You › history

  Questions this session:
   1. How do I reset my password?
   2. What if I don't get the email?
   3. What are the API rate limits for the Pro plan?
   ...

  ↑ Notice: the agent has no idea any of these were asked.
    Each answer was built from zero context.
    Stage 2 fixes this with ChromaDB episodic memory.
```

The `history` command shows every question you asked — but from the agent's
perspective, that list does not exist. This is exactly the problem Stage 2
solves by persisting resolved interactions into ChromaDB and injecting
relevant past cases into the system prompt on each new query.
