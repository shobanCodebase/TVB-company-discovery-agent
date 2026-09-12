"""
Domain-based deduplication for discovered candidates / researched companies.

Primary dedup key is normalized domain. Falls back to a normalized
company-name key when domain is missing (rare, since discovery should
always try to attach a domain).
"""

from __future__ import annotations

import re
from typing import Iterable, TypeVar

T = TypeVar("T")


def normalize_domain(domain: str) -> str:
    d = domain.strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = re.sub(r"^www\.", "", d)
    d = d.split("/")[0]
    d = d.rstrip(".")
    return d


def normalize_company_name(name: str) -> str:
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9]+", " ", n)
    n = re.sub(
        r"\b(inc|llc|ltd|limited|corp|corporation|co|gmbh|pvt|private|pte|"
        r"plc|srl|sa|bv|oy|ab)\b",
        "",
        n,
    )
    n = re.sub(r"\s+", " ", n).strip()
    return n


def dedup_key(company_name: str, domain: str | None) -> str:
    if domain:
        norm = normalize_domain(domain)
        if norm:
            return f"domain:{norm}"
    return f"name:{normalize_company_name(company_name)}"


def deduplicate(
    items: Iterable[T],
    get_company_name,
    get_domain,
) -> list[T]:
    """
    Generic dedup over any object type that exposes company_name/domain
    via the provided accessor callables. Keeps the FIRST occurrence of
    each key.

    Example:
        deduplicate(
            candidates,
            get_company_name=lambda c: c.company_name,
            get_domain=lambda c: c.domain,
        )
    """
    seen: set[str] = set()
    result: list[T] = []
    for item in items:
        key = dedup_key(get_company_name(item), get_domain(item))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result