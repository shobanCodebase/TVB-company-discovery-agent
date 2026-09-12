from core.deduplication import (
    dedup_key,
    deduplicate,
    normalize_company_name,
    normalize_domain,
)
from core.models import CompanyCandidate


def test_normalize_domain_strips_protocol_and_www():
    assert normalize_domain("https://www.Acme.io/about") == "acme.io"
    assert normalize_domain("http://acme.io") == "acme.io"
    assert normalize_domain("acme.io") == "acme.io"


def test_normalize_company_name_strips_suffixes():
    assert normalize_company_name("Acme Inc.") == "acme"
    assert normalize_company_name("Acme, LLC") == "acme"
    assert normalize_company_name("Acme GmbH") == "acme"


def test_dedup_key_prefers_domain():
    k1 = dedup_key("Acme Inc", "acme.io")
    k2 = dedup_key("Acme SaaS Platform", "www.acme.io")
    assert k1 == k2


def test_dedup_key_falls_back_to_name():
    k1 = dedup_key("Acme Inc", None)
    k2 = dedup_key("Acme, LLC", "")
    assert k1 == k2


def test_deduplicate_keeps_first_occurrence():
    candidates = [
        CompanyCandidate(company_name="Acme Inc", domain="acme.io", source_url="https://a.com"),
        CompanyCandidate(company_name="Acme SaaS", domain="www.acme.io", source_url="https://b.com"),
        CompanyCandidate(company_name="Beta Corp", domain="beta.com", source_url="https://c.com"),
    ]
    result = deduplicate(
        candidates,
        get_company_name=lambda c: c.company_name,
        get_domain=lambda c: c.domain,
    )
    assert len(result) == 2
    assert result[0].source_url == "https://a.com"
    assert result[1].company_name == "Beta Corp"


def test_deduplicate_no_domain_uses_name():
    candidates = [
        CompanyCandidate(company_name="Acme Inc", domain=None, source_url="https://a.com"),
        CompanyCandidate(company_name="Acme, LLC", domain=None, source_url="https://b.com"),
    ]
    result = deduplicate(
        candidates,
        get_company_name=lambda c: c.company_name,
        get_domain=lambda c: c.domain,
    )
    assert len(result) == 1