from core.models import CompanyCandidate
from core.relevance import filter_and_prioritize, is_relevant_candidate, relevance_score


def make_candidate(name, url, domain=None, snippet=""):
    return CompanyCandidate(
        company_name=name, domain=domain, source_url=url, search_snippet=snippet,
    )


def test_wikipedia_rejected():
    c = make_candidate("Some Topic", "https://en.wikipedia.org/wiki/Some_Topic", domain="en.wikipedia.org")
    assert is_relevant_candidate(c) is False


def test_government_site_rejected():
    c = make_candidate("Report", "https://www.cdc.gov/report/thing", domain="cdc.gov")
    assert is_relevant_candidate(c) is False


def test_university_edu_domain_rejected():
    c = make_candidate("UW Research", "https://cs.uw.edu/research/gesture", domain="cs.uw.edu")
    assert is_relevant_candidate(c) is False


def test_recipe_path_rejected():
    c = make_candidate("Cookie Recipe", "https://allrecipes.com/recipe/12345/cookies", domain="allrecipes.com")
    assert is_relevant_candidate(c) is False


def test_news_article_without_positive_signal_rejected():
    c = make_candidate(
        "Alzheimer's Facts and Figures Report",
        "https://alz.org/report/facts-and-figures",
        domain="alz.org",
        snippet="A comprehensive report on Alzheimer's statistics.",
    )
    assert is_relevant_candidate(c) is False


def test_brand_negative_term_rejected():
    c = make_candidate("Applebee's DOLLARITA", "https://applebees.com/promo/dollarita", domain="applebees.com")
    assert is_relevant_candidate(c) is False


def test_valid_startup_homepage_accepted():
    c = make_candidate(
        "Acme SaaS", "https://acme.io", domain="acme.io",
        snippet="Acme is a SaaS platform that raised $2.5M in seed funding.",
    )
    assert is_relevant_candidate(c) is True


def test_unfamiliar_domain_with_positive_signal_accepted():
    c = make_candidate(
        "Zylotech Platform", "https://zylotech.ai", domain="zylotech.ai",
        snippet="Zylotech is a fintech startup founded by Jane Doe, CEO.",
    )
    assert is_relevant_candidate(c) is True


def test_unfamiliar_domain_no_snippet_still_accepted_absent_negative_signals():
    # No positive OR negative signals -> should NOT be rejected (conservative filter)
    c = make_candidate("Unknown Startup Inc", "https://unknownstartup.io/about", domain="unknownstartup.io")
    assert is_relevant_candidate(c) is True


def test_relevance_score_ranks_homepage_above_generic_article():
    homepage = make_candidate(
        "Acme SaaS", "https://acme.io", domain="acme.io",
        snippet="Acme is a technology platform that raised $2M in funding.",
    )
    article = make_candidate(
        "Some Report on Startups", "https://news.example.com/articles/2024/startups-report",
        domain="news.example.com", snippet="A report on the state of the startup ecosystem.",
    )
    assert relevance_score(homepage) > relevance_score(article)


def test_relevance_score_ranks_garbage_very_low():
    garbage = make_candidate("Applebee's DOLLARITA", "https://applebees.com/promo", domain="applebees.com")
    startup = make_candidate(
        "Acme SaaS", "https://acme.io", domain="acme.io", snippet="Acme raised $2M seed funding.",
    )
    assert relevance_score(garbage) < relevance_score(startup)


def test_filter_and_prioritize_caps_and_orders():
    candidates = [
        make_candidate("Wiki Page", "https://en.wikipedia.org/wiki/X", domain="en.wikipedia.org"),
        make_candidate("Acme SaaS", "https://acme.io", domain="acme.io", snippet="SaaS platform, raised funding."),
        make_candidate("Beta Startup", "https://beta.com", domain="beta.com", snippet="tech startup founder CEO"),
        make_candidate("Recipe Site", "https://allrecipes.com/recipe/1", domain="allrecipes.com"),
    ]
    selected, rejected_count = filter_and_prioritize(candidates, max_candidates=1)
    assert rejected_count == 2  # wiki + recipe rejected
    assert len(selected) == 1
    assert selected[0].company_name in {"Acme SaaS", "Beta Startup"}

def test_listicle_roundup_rejected():
    c = make_candidate(
        "Top 1258 SaaS/1000 Startups 2026",
        "https://example.com/lists/top-saas-startups-2026",
        domain="example.com",
        snippet="A ranked list of the top funded SaaS startups this year.",
    )
    assert is_relevant_candidate(c) is False


def test_funding_explainer_article_rejected():
    c = make_candidate(
        "What is SaaS Seed Funding? Key Metrics & Process",
        "https://example.com/blog/saas-seed-funding-explained",
        domain="example.com",
    )
    assert is_relevant_candidate(c) is False


def test_bare_place_name_rejected():
    c = make_candidate("London", "https://example.com/hubs/london", domain="example.com")
    assert is_relevant_candidate(c) is False


def test_bare_category_name_rejected():
    c = make_candidate("Y Combinator", "https://ycombinator.com", domain="ycombinator.com")
    assert is_relevant_candidate(c) is False


def test_real_company_mentioning_london_still_accepted():
    c = make_candidate(
        "Acme Fintech", "https://acmefintech.io", domain="acmefintech.io",
        snippet="Acme Fintech, based in London, raised $2M in seed funding.",
    )
    assert is_relevant_candidate(c) is True

def test_how_to_raise_article_rejected():
    c = make_candidate(
        "How to Raise a Seed Round for an AI Startup (2026)",
        "https://example.com/blog/how-to-raise-seed", domain="example.com",
    )
    assert is_relevant_candidate(c) is False


def test_state_of_startups_report_rejected():
    c = make_candidate(
        "2025 State of Startups in the Southeast",
        "https://example.com/reports/2025-state-of-startups", domain="example.com",
    )
    assert is_relevant_candidate(c) is False


def test_bare_category_word_rejected():
    c = make_candidate("Investors", "https://example.com/investors", domain="example.com")
    assert is_relevant_candidate(c) is False


def test_bare_sector_word_rejected():
    c = make_candidate("Healthtech", "https://example.com/sectors/healthtech", domain="example.com")
    assert is_relevant_candidate(c) is False

def test_vc_fund_by_domain_rejected():
    c = make_candidate("Look AI Ventures", "https://lookai.vc", domain="lookai.vc")
    assert is_relevant_candidate(c) is False


def test_vc_fund_by_name_rejected():
    c = make_candidate("Pear VC", "https://pear.vc/portfolio", domain="pear.vc")
    assert is_relevant_candidate(c) is False


def test_accelerator_rejected():
    c = make_candidate("Founder Institute, World's Largest AI", "https://fi.co", domain="fi.co")
    assert is_relevant_candidate(c) is False


def test_startup_catalyst_program_rejected():
    c = make_candidate("IFC Startup Catalyst", "https://ifc.org/startup-catalyst", domain="ifc.org")
    assert is_relevant_candidate(c) is False


def test_seed_fund_list_rejected():
    c = make_candidate("Funds that lead seeds", "https://vcsheet.com/sheet/funds-that-lead-seeds", domain="vcsheet.com")
    assert is_relevant_candidate(c) is False


def test_portfolio_company_mentioning_investor_still_accepted():
    c = make_candidate(
        "Acme SaaS raises $2M from Sequoia Ventures",
        "https://acme.io", domain="acme.io",
        snippet="Acme SaaS, a technology platform, raises $2M seed funding led by Sequoia Ventures.",
    )
    assert is_relevant_candidate(c) is True