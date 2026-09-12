"""
Deterministic candidate relevance pre-filter + priority scoring.

Runs AFTER discovery + deduplication, BEFORE deep research (Groq calls).
This module NEVER decides qualification — core.validators.qualify() is
the only qualification authority. It only decides:
  (a) is this candidate plausible enough as a real company to be worth
      the cost of an LLM extraction call, and
  (b) if we can only afford to research N candidates this round, which
      ones should go first.

A real but unfamiliar startup must always survive this filter — it is
intentionally conservative about rejecting, using several independent
negative signals (domain, URL path, title/snippet wording) rather than
a single blunt blacklist.
"""

from __future__ import annotations

from core.models import CompanyCandidate

NEGATIVE_PATH_MARKERS = (
    "/wiki/", "/recipe/", "/recipes/", "/pdf/", "/report/", "/reports/",
    "/article/", "/articles/", "/news/", "/blog/", "/dictionary/",
    "/define/", "/definition/", "/jobs/", "/careers/", "/job/",
    "/docs/", "/documentation/", "/dataset/", "/datasets/", "/research/",
    "/paper/", "/papers/", "/menu/", "/restaurants/", "/restaurant/",
)

NEGATIVE_DOMAINS = (
    "wikipedia.org", "wiktionary.org", "investopedia.com", "britannica.com",
    "allrecipes.com", "foodnetwork.com", "epicurious.com", "food.com",
    ".gov", ".edu", "nih.gov", "cdc.gov", "who.int",
    "youtube.com", "reddit.com", "quora.com", "pinterest.com",
    "amazon.com", "ebay.com", "walmart.com",
    "glassdoor.com", "indeed.com",
)

NEGATIVE_BRAND_TERMS = (
    "applebee", "raising cane", "mcdonald", "starbucks", "chipotle",
    "dollarita", "gesture recognizer", "facts and figures",
)

# VC firms, accelerators, and funds are investors/programs, not the
# portfolio companies we're trying to qualify. A search result whose own
# name/snippet centers on ITS OWN identity as a fund/accelerator (not a
# mention of who invested in some other company) should be filtered.
NEGATIVE_INVESTOR_DOMAIN_MARKERS = (
    ".vc", "ventures.com", "capital.com", "-capital", "vcsheet.com",
)

NEGATIVE_INVESTOR_NAME_TERMS = (
    "ventures", "capital", "accelerator", "incubator", "seed fund",
    "venture capital", "vc firm", "angel investors", "startup catalyst",
    "founder institute", "y combinator", "techstars",
)

POSITIVE_TERMS = (
    "startup", "platform", "saas", "software", "technology", "tech ",
    "app", "api", "cloud", "fintech", "healthtech", "edtech",
    "raises", "raised", "funding", "seed round", "series a",
    "co-founder", "cofounder", "founder", "ceo",
)

NEGATIVE_TITLE_TERMS = (
    "recipe", "how to", "definition", "wikipedia", "dictionary",
    "facts and figures", "report", "study finds", "research paper",
    "job opening", "hiring", "vacancy", "menu", "restaurant chain",
)

# Listicle/roundup/explainer pages that talk ABOUT startups/funding in
# aggregate, rather than being a page about one specific company. These
# contain plenty of "positive" terminology (funding, SaaS, startup) so
# NEGATIVE_TITLE_TERMS alone doesn't catch them.
LISTICLE_MARKERS = (
    "top ", "recently funded", "benchmarks", "key metrics", "trajectories",
    "last 30 days", "updated weekly", "stages:", "what is", "rebound",
    "requirements", "funded startups", "funded transportation",
    "funding rounds", "funding trends", " vs ", "best ", "list of",
    "how to raise", "how to start", "fundraising trends", "state of startups",
    "state of the", "funded b2b", "funded saas", "after the ", "crash",
    "trends 2024", "trends 2025", "trends 2026",
)

# Bare place/category/sector names with no company-specific content — a
# search result whose entire title IS one of these (exact match, not
# substring) is a hub/tag/category page, not a company. A real company
# merely mentioning one of these words elsewhere in its name or snippet
# is unaffected, since this only matches the full company_name.
GENERIC_STANDALONE_NAMES = {
    "london", "saudi arabia", "seed fund", "pre", "financing & investment",
    "y combinator", "seed funding", "series a", "series b",
    "investors", "healthtech", "fintech", "edtech", "saas", "startups",
    "america's seed fund", "moroccan e",
}

CONTENT_TAGS_PRIORITY_UNUSED = None  # (placeholder removed — not used here)


def _domain_is_negative(domain: str | None) -> bool:
    if not domain:
        return False
    d = domain.lower()
    return any(marker in d for marker in NEGATIVE_DOMAINS)


def _path_is_negative(url: str) -> bool:
    return any(marker in url.lower() for marker in NEGATIVE_PATH_MARKERS)


def _text_blob(candidate: CompanyCandidate) -> str:
    return " ".join(filter(None, [candidate.company_name, candidate.search_snippet or ""])).lower()


def _positive_term_count(text: str) -> int:
    return sum(1 for term in POSITIVE_TERMS if term in text)


def _has_negative_title_terms(text: str) -> bool:
    return any(term in text for term in NEGATIVE_TITLE_TERMS)


def _has_negative_brand_terms(text: str) -> bool:
    return any(term in text for term in NEGATIVE_BRAND_TERMS)


def _is_listicle(text: str) -> bool:
    return any(marker in text for marker in LISTICLE_MARKERS)


def _is_generic_standalone_name(company_name: str) -> bool:
    return company_name.strip().lower() in GENERIC_STANDALONE_NAMES


def _is_investor_or_accelerator(candidate: CompanyCandidate) -> bool:
    """
    Rejects candidates that ARE a VC fund, accelerator, or incubator
    (their own identity, per domain and/or name), as opposed to a
    portfolio company that happens to mention an investor by name (e.g.
    "Acme raises $2M from Sequoia" should NOT be rejected here — only
    "Sequoia Capital" or "Pear VC" as the subject itself should be).
    """
    domain = (candidate.domain or "").lower()
    name = candidate.company_name.strip().lower()

    if any(marker in domain for marker in NEGATIVE_INVESTOR_DOMAIN_MARKERS):
        return True

    # Name-level check: only trip if an investor term appears WITHOUT any
    # accompanying funding/startup-of-someone-else language, to avoid
    # false-rejecting "Startup X, backed by XYZ Ventures, raises $2M".
    if any(term in name for term in NEGATIVE_INVESTOR_NAME_TERMS):
        # If the name also contains a funding-recipient signal like
        # "raises"/"secures"/"$", it's more likely a portfolio company
        # story that just happens to mention the investor's name.
        if not any(sig in name for sig in ("raises", "secures", "closes", "$", "funding round")):
            return True

    return False


def is_relevant_candidate(candidate: CompanyCandidate) -> bool:
    """
    Cheap deterministic pass/fail. Conservative by design: a candidate is
    only rejected on a clear negative signal, never for being unfamiliar.
    """
    text = _text_blob(candidate)

    if _domain_is_negative(candidate.domain):
        return False
    if _path_is_negative(candidate.source_url):
        return False
    if _has_negative_brand_terms(text):
        return False
    if _has_negative_title_terms(text) and _positive_term_count(text) == 0:
        return False
    if _is_listicle(text):
        return False
    if _is_generic_standalone_name(candidate.company_name):
        return False
    if _is_investor_or_accelerator(candidate):
        return False

    return True


def relevance_score(candidate: CompanyCandidate) -> int:
    """Priority score for ordering — NOT a qualification score."""
    text = _text_blob(candidate)
    score = _positive_term_count(text) * 5

    if candidate.domain and not _domain_is_negative(candidate.domain):
        if candidate.source_url.rstrip("/").count("/") <= 3:
            score += 10

    if _has_negative_title_terms(text):
        score -= 15
    if _has_negative_brand_terms(text):
        score -= 30
    if len(candidate.company_name.strip()) < 3:
        score -= 10

    return score


def filter_and_prioritize(
    candidates: list[CompanyCandidate],
    max_candidates: int,
) -> tuple[list[CompanyCandidate], int]:
    """
    Applies the relevance filter, ranks survivors by priority score
    (descending), and caps to max_candidates.

    Returns (selected_candidates, rejected_count).
    """
    relevant = [c for c in candidates if is_relevant_candidate(c)]
    rejected_count = len(candidates) - len(relevant)
    ranked = sorted(relevant, key=relevance_score, reverse=True)
    return ranked[:max_candidates], rejected_count