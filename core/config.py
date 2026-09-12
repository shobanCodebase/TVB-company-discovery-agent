"""
Centralized configuration loaded from environment variables.

All secrets (API keys) come from env vars only — never hardcoded.
Locally these come from a .env file (via python-dotenv); on Streamlit
Community Cloud they come from st.secrets, which app.py mirrors into
os.environ before this module's Settings() is constructed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- API keys ---
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    search_api_key: str = field(default_factory=lambda: os.getenv("SEARCH_API_KEY", ""))
    search_provider: str = field(default_factory=lambda: os.getenv("SEARCH_PROVIDER", "tavily"))
    hunter_api_key: str = field(default_factory=lambda: os.getenv("HUNTER_API_KEY", ""))

    # --- LLM ---
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-oss-20b"))
    llm_temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.1))
    llm_extraction_max_tokens: int = field(
        default_factory=lambda: _get_int("LLM_EXTRACTION_MAX_TOKENS", 900)
    )
    llm_query_gen_max_tokens: int = field(
        default_factory=lambda: _get_int("LLM_QUERY_GEN_MAX_TOKENS", 300)
    )
    groq_min_request_interval_seconds: float = field(
        default_factory=lambda: _get_float("GROQ_MIN_REQUEST_INTERVAL_SECONDS", 20.0)
    )

    # --- Research input sizing (controls LLM token usage) ---
    research_max_source_chars: int = field(
        default_factory=lambda: _get_int("RESEARCH_MAX_SOURCE_CHARS", 5000)
    )

    # --- Pipeline tuning ---
    target_qualified_leads: int = field(default_factory=lambda: _get_int("TARGET_QUALIFIED_LEADS", 20))
    min_qualified_leads: int = field(default_factory=lambda: _get_int("MIN_QUALIFIED_LEADS", 15))
    max_discovery_queries: int = field(default_factory=lambda: _get_int("MAX_DISCOVERY_QUERIES", 25))
    max_candidates_per_query: int = field(default_factory=lambda: _get_int("MAX_CANDIDATES_PER_QUERY", 8))
    max_candidates_total: int = field(default_factory=lambda: _get_int("MAX_CANDIDATES_TOTAL", 150))
    max_candidates_for_research: int = field(
        default_factory=lambda: _get_int("MAX_CANDIDATES_FOR_RESEARCH", 20)
    )
    max_research_iterations: int = field(default_factory=lambda: _get_int("MAX_RESEARCH_ITERATIONS", 60))
    request_timeout_seconds: float = field(default_factory=lambda: _get_float("REQUEST_TIMEOUT_SECONDS", 15.0))
    max_retries: int = field(default_factory=lambda: _get_int("MAX_RETRIES", 2))

    # --- Financial criteria ---
    min_amount_usd: float = field(default_factory=lambda: _get_float("MIN_AMOUNT_USD", 1_000_000))
    max_amount_usd: float = field(default_factory=lambda: _get_float("MAX_AMOUNT_USD", 5_000_000))

    # --- Feature flags ---
    use_email_verification_api: bool = field(default_factory=lambda: _get_bool("USE_EMAIL_VERIFICATION_API", True))
    enable_cache: bool = field(default_factory=lambda: _get_bool("ENABLE_CACHE", True))

    def validate_required(self) -> list[str]:
        """Returns a list of missing required keys (does not raise)."""
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.search_api_key:
            missing.append("SEARCH_API_KEY")
        return missing


settings = Settings()