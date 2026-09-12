# TVB Company Discovery Agent

An autonomous agent that discovers technology companies matching a strict, evidence-backed
lead-qualification profile — live from the public web, with no hardcoded company list — and
outputs verified CEO/co-founder contact emails for outreach.

Built for The Venture Build (TVB) Agentic & Automation Intern take-home assignment.

## Problem

Manually finding companies that fit a narrow, multi-criteria profile — a specific funding/revenue
band, a technology platform, limited US presence, an identifiable founder, and a real professional
email for that person — is slow, error-prone, and hard to keep evidence-backed at scale. Doing it
well requires searching diverse live sources, extracting facts without guessing, and verifying
contact details rather than assuming a plausible-looking email works.

## Solution

The agent dynamically discovers technology companies from live web sources — never from a fixed or
hardcoded list — and evaluates each one against the take-home's hard qualification criteria:

- Revenue **or** funding between **$1M and $5M USD**
- A **technology-related platform**
- **Minimal to no US presence**
- A **named CEO or co-founder**
- A **verified professional email** for that person
- Every factual claim backed by **retained evidence** (source URL + excerpt)
- **Deterministic AND qualification** — all criteria must pass; none is optional and none is
  inferred from partial information

## Architecture
```bash
Discovery (dynamic, multi-query, live search)
↓
Relevance filtering (deterministic pre-filter — rejects non-companies before expensive research)
↓
Deduplication (domain-based, across candidates and across discovery rounds)
↓
Web research (fetches public pages + targeted search for each candidate)
↓
Structured LLM extraction (funding, tech platform, US presence, HQ, contact — each with evidence)
↓
Evidence validation (rejects any claim whose excerpt isn't actually found in the fetched source)
↓
Deterministic qualification (plain Python — the LLM has no vote here)
↓
Contact discovery (pattern generation + optional Hunter.io lookup)
↓
Email verification (syntax, domain match, MX record, optional confidence score)
↓
Final lead output (Streamlit UI, CSV/JSON export)
```


**The LLM assists with extraction and reasoning — it does not make the final qualification
decision.** Every mandatory criterion (financial range, tech platform, US presence, named contact,
verified email) is checked by deterministic Python logic in `core/validators.py`. A candidate the
LLM believes is promising still gets rejected if the deterministic checks and their required
evidence aren't satisfied.

## Evidence-First Design

This is the core strength of the project, and the reason its qualification decisions can be
trusted rather than merely plausible:

- Every factual claim (funding amount, tech platform, US presence, contact identity) must come
  with an `Evidence` object: a claim, the source URL it was fetched from, and a short excerpt.
- Before that evidence is accepted, the pipeline checks that the excerpt actually appears in the
  real page text fetched from that URL. An LLM claim that isn't backed by the actual source content
  is discarded — not trusted, not softened, discarded.
- If a field cannot be established with evidence, it is treated as `UNKNOWN`, never guessed or
  inferred. For mandatory fields, `UNKNOWN` means that criterion fails.
- Deterministic validators — not the LLM — make the final pass/fail call on every mandatory
  criterion, using hard AND logic. A display-only score (0–100) exists purely for ranking
  already-qualified leads in the UI; it never compensates for a failed mandatory criterion.

## Dynamic Discovery

Companies are discovered **dynamically, from live search results, on every run** — there is no
hardcoded company list anywhere in this codebase, and none is used as a substitute for live
evidence. Discovery generates diverse search queries from a deterministic template pool (rotated
across discovery rounds so repeated runs don't just re-search the same queries), optionally
supplemented by LLM-generated queries for additional angle diversity. A deterministic relevance
pre-filter then screens out obvious non-companies (news aggregators, listicles, government/academic
pages, VC funds and accelerators, generic category pages) before any candidate reaches the more
expensive research and extraction stage — this improves precision and reduces wasted API calls
without ever deciding qualification itself; that decision stays entirely in the deterministic
validators described above.

## Qualification Logic

A lead is qualified **only if all of the following pass** (hard AND — a display score never
compensates for a failed mandatory check):
```bash
Financial (funding OR revenue between $1M–$5M, with evidence)
AND Tech platform (with evidence)
AND US presence in {NONE, LOW} (UNKNOWN is rejected, not assumed safe)
AND Named CEO or co-founder (not "team"/"management", with evidence)
AND Verified professional email (not info@/hello@/contact@/etc.)
```


**Important distinction:** the system *targets* at least 15 qualifying leads per run, but the final
lead count depends on the live evidence available during that specific run and on external API
quotas at the time. **The agent intentionally returns fewer leads rather than relaxing mandatory
criteria or fabricating missing evidence.** A run producing 0, 3, or 12 qualified leads with
transparent, evidence-backed rejection reasons for every non-qualifying candidate reflects the
system working correctly, not a malfunction — see Limitations and Near-Miss Behavior below.

## Near-Miss Behavior

Because every mandatory criterion requires its own independent evidence, a company can genuinely
satisfy most of the profile and still be excluded because one specific fact wasn't available in the
sources the pipeline happened to fetch that run. For example, observed patterns from real runs
include companies with:

- Qualifying funding, a real technology platform, **and** low US presence — but excluded because no
  named CEO or co-founder appeared in the available source text.
- A named contact and a qualifying technology platform — but excluded because the specific funding
  amount wasn't clearly stated in that run's fetched sources.
- A qualifying funding amount and a named contact — but excluded because no email meeting the
  verification bar (not a generic address, passing syntax/domain/MX checks) could be found.

This is intentional, not a bug: the system does not fill a missing mandatory fact with an inference
or a plausible guess just to push a near-miss company over the line. Other rejections are
straightforwardly correct rather than near-misses — for instance, a company whose funding round is
well outside the $1M–$5M band, or whose US presence is clearly substantial, is rejected on the
merits, exactly as intended.

## Anti-Hallucination / Evidence Strategy

See "Evidence-First Design" above — in short: claims without a verifiable excerpt from a real
fetched page are discarded before they can ever influence a qualification decision, and unknown
fields are never guessed.

## Tech Stack

- Python 3.11+
- Streamlit — UI
- Pydantic — evidence-first data models
- httpx — HTTP client for search API, web fetching, and email verification calls
- BeautifulSoup4 — HTML parsing/text extraction
- Groq API — structured extraction and query generation (model configurable via `.env`)
- Tavily Search API — live web search (called directly via `httpx`, no SDK dependency)
- Hunter.io API (optional) — professional email finding and verification
- dnspython — MX record lookups for email domain validation
- pandas — results table formatting in the UI
- pytest, tenacity — testing and retry handling

## Project Structure
```bash
tvb-agentic-company-discovery/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── .streamlit/
│ └── config.toml
├── core/
│ ├── models.py
│ ├── config.py
│ ├── validators.py
│ ├── deduplication.py
│ ├── scoring.py
│ ├── relevance.py
│ └── pipeline.py
├── agents/
│ ├── discovery_agent.py
│ ├── research_agent.py
│ └── contact_agent.py
├── services/
│ ├── search_service.py
│ ├── web_service.py
│ ├── llm_service.py
│ └── email_service.py
└── tests/
└── (unit tests for every module above)
```


## Setup (Windows)

```powershell
cd /d "C:\path\to\tvb-agentic-company-discovery"

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

copy .env.example .env
```

Then open `.env` and fill in your own API keys (see below). **`.env` is for local development
only — it is listed in `.gitignore` and must never be committed.**

## Environment Variables

**Required for a real (non-test) run:**

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | LLM extraction and query generation |
| `SEARCH_API_KEY` | Live web search (Tavily) |

**Optional (recommended for better precision):**

| Variable | Purpose | Behavior if unset |
|---|---|---|
| `HUNTER_API_KEY` | Email finding + verification confidence score | Falls back to pattern generation + deterministic checks (syntax/domain/MX) only |

**Performance / rate-limit tuning:**

| Variable | Default | Purpose |
|---|---|---|
| `MAX_CANDIDATES_FOR_RESEARCH` | `20` | Caps how many relevance-filtered candidates are sent to the (expensive) LLM research stage per discovery round |
| `GROQ_MIN_REQUEST_INTERVAL_SECONDS` | `20` | Proactively spaces out actual Groq API calls to reduce how often a per-minute (TPM) rate limit is hit in the first place |
| `RESEARCH_MAX_SOURCE_CHARS` | `5000` | Caps how much combined page text is sent to the LLM per candidate |
| `LLM_EXTRACTION_MAX_TOKENS` | `900` | Output token budget for the structured extraction call |

Proactive throttling (`GROQ_MIN_REQUEST_INTERVAL_SECONDS`) trades runtime for reliability: spacing
calls out substantially reduces Groq TPM-related failures, at the cost of a run taking longer than
it would with unthrottled, back-to-back calls. Targeted Tavily searches per candidate were reduced
to one (covering funding, tech platform, US presence, and contact in a single combined query) to
further control external API consumption. Actual runtime varies with live site response times, API
rate-limit behavior on a given day, and how many candidates survive the relevance filter — this
project does not claim or target a fixed runtime.

**Other tunable settings** (sensible defaults ship in `.env.example`): `SEARCH_PROVIDER`,
`LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_QUERY_GEN_MAX_TOKENS`, `TARGET_QUALIFIED_LEADS`,
`MIN_QUALIFIED_LEADS`, `MAX_DISCOVERY_QUERIES`, `MAX_CANDIDATES_PER_QUERY`,
`MAX_CANDIDATES_TOTAL`, `MAX_RESEARCH_ITERATIONS`, `REQUEST_TIMEOUT_SECONDS`, `MAX_RETRIES`,
`MIN_AMOUNT_USD`, `MAX_AMOUNT_USD`, `USE_EMAIL_VERIFICATION_API`, `ENABLE_CACHE`.

**Test-only / no key required:** the entire `pytest` suite runs with zero real API keys — every
external call (Tavily, Groq, Hunter, DNS, HTTP fetches) is mocked in the tests.

## Local Run

```powershell
streamlit run app.py
```

Opens at `http://localhost:8501`. Click **RUN DISCOVERY** to start a fresh, live pipeline run — no
cached, bundled, or hardcoded results are ever used; every click performs new discovery, research,
and qualification from current live sources.

## Testing

```powershell
pytest -q
```

Current local validation: **191 tests passing**, 0 failed, all using mocked external services (no
real API keys required to run the suite).

## Streamlit Community Cloud Deployment

1. Push this repository to GitHub (public, since the evaluator needs to open it without cloning).
2. Go to [Streamlit Community Cloud](https://share.streamlit.io) and sign in.
3. Click **New app** and select this repository.
4. Select the branch you want deployed (e.g. `main`).
5. Set the entry point file to `app.py`.
6. Before deploying, open **Advanced settings → Secrets** and configure your secrets there (see
   below) — **do not commit any API keys to the repository**.
7. Click **Deploy**.
8. Once the build finishes, open the public URL Streamlit gives you.
9. Click **RUN DISCOVERY** in the deployed app to confirm it performs a genuine, fresh discovery run
   — the deployed instance behaves identically to a local run: no hardcoded lead list, no bundled
   demo data, live sources queried on demand.

## Streamlit Secrets

In the Streamlit Cloud **Secrets** panel, paste TOML in this shape (replace with your own real
values — these are placeholders only, never commit actual keys anywhere in this repository):

```toml
GROQ_API_KEY = "your-groq-key-here"
SEARCH_API_KEY = "your-tavily-key-here"
HUNTER_API_KEY = "your-hunter-key-here"
```

`HUNTER_API_KEY` may be omitted if you don't have one — the app degrades gracefully (see
Environment Variables above). Locally, the same values go in `.env` (via `.env.example` as a
template); on Streamlit Cloud, they're read from this Secrets panel and mirrored into the
environment at app startup — both paths feed the same configuration code, so behavior is identical.

## Security

- No API keys are hardcoded anywhere in the source.
- `.env` is listed in `.gitignore` and is never committed; it exists for local development only.
- Locally, keys are read from `.env` via `python-dotenv`; on Streamlit Cloud, they're read from the
  Secrets manager.
- The UI never displays raw API key values; missing required keys are reported by name only, never
  their values.

## Limitations

This system depends on external live APIs and public web sources, so results legitimately vary
between runs — this is an inherent property of a live, evidence-based discovery agent, not a
shortcoming to work around:

- **API rate limits and quotas.** Groq's account-tier TPM (tokens-per-minute) and daily token
  limits can restrict how many companies can actually be researched within a single run. Tavily's
  plan/request limits can similarly restrict live search volume mid-run. Both are handled
  gracefully (bounded retries, a clean stop with an explanatory message, no infinite retry loops),
  but they are a real ceiling on same-day throughput, not something the application can create more
  of.
- **Source accessibility.** HTTP 403/401/429 responses, SSL/timeout failures, and paywalled or
  bot-blocked pages reduce the evidence available for any given candidate. The pipeline skips these
  gracefully and continues with whatever sources are accessible.
- **Evidence gaps in source material.** Some articles report a funding round without naming a
  CEO or co-founder. Some name a founder without providing any discoverable, verifiable professional
  email. Some report an amount in ambiguous terms the extraction step correctly refuses to guess at.
  Each of these is a genuine gap in the public evidence available for that company at that time —
  not a defect in extraction.
- **Email verification limits.** Verification quality depends on whether an external verification
  service (Hunter.io) is configured and has remaining quota; without one, the system still verifies
  syntax, domain match, and MX records, which is a real but lower-precision signal.
- **Consequently, a single run can legitimately produce fewer than 15 qualified leads even when
  every component of the application is functioning correctly.** The system targets at least 15
  qualifying leads per run, but the final count depends on the live evidence available during that
  run and on external API quotas at the time — the agent intentionally returns fewer leads rather
  than relaxing mandatory criteria or fabricating missing evidence to reach a target number.
- Non-English source pages are not specifically handled — extraction quality may degrade where the
  only public information about a company is in a language other than English.

## Future Improvements

- Persist a cross-run dedup ledger (beyond the current in-run cache) so repeated evaluator runs
  don't re-research the same companies.
- Add non-English query variants for broader international coverage.
- Add a second search provider as a fallback if Tavily's quota is exhausted mid-run.
- Add a second LLM provider as a fallback if the primary provider's quota is exhausted mid-run.

## Why This Matters for TVB

TVB is an AI-powered venture catalyst ecosystem focused on helping ambitious startups and
scale-ups grow through execution support, market access, ecosystem connections, and capital
readiness. Sourcing companies that fit a specific growth-stage and platform profile is a natural fit
for this kind of automated, evidence-backed discovery agent. This context explains why the project
is relevant to TVB's work; it is not a source of qualification criteria, and no company mentioned in
TVB's own reference or example materials is treated as a qualifying lead or used as a discovery
seed — every lead this system produces is validated purely against the take-home's own criteria
using live, independently fetched public evidence.