"""Settings (CONTRACT.md: Settings): defaults, .env override, environment precedence, types."""

import pydantic
import pytest
from app import config
from app.config import Settings, get_settings

FIELDS = list(Settings.model_fields)

CONTRACT_FIELDS = {
    "kit_base_url", "kit_participant_key", "anthropic_api_key", "llm_model", "llm_effort",
    "llm_max_tokens", "deepgram_api_key", "dg_stt_model", "dg_tts_model", "gemini_api_key",
    "embed_model", "embed_dim", "database_url", "doc_mode", "doc_context_token_budget",
    "doc_refresh_seconds", "app_host", "app_port", "cookie_secure", "public_origin",
    "session_ttl_hours",
}


@pytest.fixture
def clean_env(monkeypatch):
    for name in FIELDS:
        monkeypatch.delenv(name.upper(), raising=False)
    return monkeypatch


def test_settings_has_every_contract_field():
    assert CONTRACT_FIELDS <= set(FIELDS)


def test_defaults_without_env_file(clean_env):
    s = Settings(_env_file=None)
    assert s.llm_max_tokens == 4096
    assert s.embed_model == "" and s.embed_dim == 768
    assert s.session_ttl_hours == 8
    assert s.cookie_secure is False
    assert s.doc_mode == "auto"
    assert s.doc_context_token_budget == 60000
    assert s.kit_base_url == "https://ht.knowvally.com"
    assert s.app_host == "127.0.0.1"  # never listen publicly by default
    for key in ("kit_participant_key", "anthropic_api_key", "deepgram_api_key", "gemini_api_key"):
        assert getattr(s, key) == ""


def test_env_file_overrides_defaults(clean_env, tmp_path):
    env = tmp_path / ".env"
    env.write_text("COOKIE_SECURE=true\nSESSION_TTL_HOURS=2\nPUBLIC_ORIGIN=https://x.example\n"
                   "DOC_MODE=rag\nUNRELATED_KEY=ignored\n# comment\n")
    s = Settings(_env_file=env)
    assert s.cookie_secure is True and s.session_ttl_hours == 2
    assert s.public_origin == "https://x.example" and s.doc_mode == "rag"
    assert not hasattr(s, "unrelated_key")


def test_environment_beats_env_file(clean_env, tmp_path):
    env = tmp_path / ".env"
    env.write_text("APP_PORT=1111\n")
    clean_env.setenv("APP_PORT", "2222")
    assert Settings(_env_file=env).app_port == 2222


@pytest.mark.parametrize("value,expected", [("true", True), ("1", True), ("yes", True),
                                            ("false", False), ("0", False), ("no", False)])
def test_cookie_secure_parsing(clean_env, value, expected):
    clean_env.setenv("COOKIE_SECURE", value)
    assert Settings(_env_file=None).cookie_secure is expected


@pytest.mark.parametrize("name,value", [("APP_PORT", "abc"), ("EMBED_DIM", "x"),
                                        ("COOKIE_SECURE", "maybe"),
                                        ("SESSION_TTL_HOURS", "1.5h")])
def test_invalid_values_rejected(clean_env, name, value):
    clean_env.setenv(name, value)
    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_env_file_is_repo_root():
    assert config.ROOT / ".env" == Settings.model_config["env_file"]
    assert (config.ROOT / "backend" / "app" / "config.py").exists()


def test_repo_env_provides_required_secrets(clean_env):
    """The real .env must set the keys the app needs (values are never printed)."""
    path = config.ROOT / ".env"
    if not path.exists():
        pytest.skip("no .env in this checkout")
    s = Settings(_env_file=path)
    missing = [k for k in ("kit_participant_key", "anthropic_api_key", "deepgram_api_key",
                           "gemini_api_key") if not getattr(s, k)]
    assert missing == []


def test_env_example_lists_every_setting_read_from_env():
    example = (config.ROOT / ".env.example").read_text()
    names = {line.split("=")[0] for line in example.splitlines() if "=" in line}
    required = {"KIT_PARTICIPANT_KEY", "ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY", "GEMINI_API_KEY",
                "DATABASE_URL", "COOKIE_SECURE", "PUBLIC_ORIGIN"}
    assert required <= names
    for line in example.splitlines():
        if line.endswith("_KEY=") or "API_KEY" in line:
            assert line.split("=", 1)[1] == ""  # the example never carries a real key
