"""Application settings, read from the environment and the repo-root .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed settings; environment variables override the .env file."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    kit_base_url: str = "https://ht.knowvally.com"
    kit_participant_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-5-5"
    llm_effort: str = "low"
    llm_max_tokens: int = 4096
    llm_keep_warm: bool = True  # keep the prompt cache warm in the background
    llm_thinking: str = "adaptive"  # adaptive | disabled (disabled: faster first text)
    history_window: int = 20  # AGT7: at most this many messages go to the model
    moderation_llm: bool = True  # MOD1: also screen user messages with a model classifier
    deepgram_api_key: str = ""
    dg_stt_model: str = "nova-3"
    dg_tts_model: str = "aura-2-thalia-en"
    gemini_api_key: str = ""
    embed_model: str = ""
    embed_dim: int = 768
    database_url: str = "postgresql://localhost:5432/kit_assistant"
    doc_mode: str = "auto"
    doc_context_token_budget: int = 60000
    doc_refresh_seconds: int = 120
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    cookie_secure: bool = False
    public_origin: str = "http://localhost:8000"
    session_ttl_hours: int = 8
    # Abuse limits (ACCEPTANCE.md ABU1-ABU10), enforced in app/limits.py. IP limits apply to
    # visitors only; logged-in users are limited per login id.
    anon_sessions_per_hour: int = 30  # ABU4, per visitor IP
    http_limit_per_minute: int = 120  # ABU5, per visitor IP or per login id
    login_limit_per_minute: int = 30  # ABU5, login attempts per IP
    login_failures_per_id: int = 5  # ABU5
    login_failure_window_seconds: int = 900
    limit_ws_per_ip: int = 3  # ABU1, visitors only
    limit_ws_per_user: int = 2  # ABU1, per login id
    limit_ws_total: int = 40
    limit_visitor_turns_per_ip_per_day: int = 20  # ABU2
    limit_user_turns_per_day: int = 100  # ABU2, per login id
    limit_utterance_seconds: int = 30  # ABU3 / ABU10
    limit_idle_seconds: int = 180
    limit_visitor_socket_seconds: int = 900
    limit_user_socket_seconds: int = 3600
    limit_pending_per_conversation: int = 3  # ABU6
    limit_admissions_per_session_per_day: int = 2
    limit_admissions_per_ip_per_day: int = 5
    limit_turns_per_conversation: int = 200  # ABU7
    limit_daily_turns: int = 5000  # ABU8, visitors stop here
    limit_daily_turns_users: int = 8000  # ABU8, logged-in users stop here
    limit_typed_chars: int = 1000  # ABU10
    limit_http_body_bytes: int = 65536
    limit_ws_message_bytes: int = 8192
    limit_audio_frame_bytes: int = 32768
    voice_gate_enabled: bool = False  # ABU11 server speech gate; set true in production
    limit_voice_seconds_visitor_ip_per_day: int = 900  # ABU11, audio sent to Deepgram
    limit_voice_seconds_user_per_day: int = 3600  # ABU11, per login id


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
