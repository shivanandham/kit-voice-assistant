"""UI_VARIANT switch: "classic" serves frontend/*.html; "app" serves the built UI (frontend/app/)
for every page, with the same session cookie, the /assistant login redirect and security headers."""

from pathlib import Path

import pytest
from app.config import get_settings

from tests.fakes.college import login_as
from tests.test_security_http import assert_security_headers

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP_INDEX = (FRONTEND / "app" / "index.html").read_text()
PAGES = {"/": "index.html", "/login": "login.html", "/history": "history.html"}


@pytest.fixture
def ui_app(monkeypatch):
    monkeypatch.setattr(get_settings(), "ui_variant", "app")


def test_default_is_classic():
    assert type(get_settings()).model_fields["ui_variant"].default == "classic"


async def test_classic_serves_the_html_pages(app_client):
    for path, name in PAGES.items():
        r = await app_client.get(path)
        assert r.status_code == 200
        assert r.text == (FRONTEND / name).read_text()


async def test_app_serves_built_index_for_every_page(ui_app, browser):
    client = await browser()
    for path in PAGES:
        r = await client.get(path)
        assert r.status_code == 200, path
        assert r.text == APP_INDEX
        assert_security_headers(r)
    await login_as(client, "arjun")
    r = await client.get("/assistant")
    assert r.status_code == 200 and r.text == APP_INDEX
    assert_security_headers(r)


async def test_app_page_sets_session_cookie(ui_app, browser):
    client = await browser()
    r = await client.get("/")
    assert "kit_session=" in r.headers.get("set-cookie", "")


async def test_app_assistant_redirects_visitors_with_cookie(ui_app, browser):
    client = await browser()
    r = await client.get("/assistant")
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert "kit_session=" in r.headers.get("set-cookie", "")
    assert_security_headers(r)


async def test_app_assets_served_and_unknown_404(ui_app, app_client):
    js = next((FRONTEND / "app").glob("index-*.js")).name
    r = await app_client.get("/ui-assets/" + js)
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    assert_security_headers(r)
    r = await app_client.get("/ui-assets/pcm-worklet.js")
    assert r.status_code == 200 and "registerProcessor" in r.text
    r = await app_client.get("/ui-assets/nope.js")
    assert r.status_code == 404
    assert_security_headers(r)
    r = await app_client.get("/ui-assets/../../backend/app/config.py")
    assert r.status_code == 404


def test_built_index_has_no_inline_script_or_external_urls():
    import re

    assert not re.search(r"<script(?![^>]*\bsrc\s*=)[^>]*>", APP_INDEX)
    assert not re.search(r"\son[a-z]+\s*=", APP_INDEX)
    assert not re.search(r"(src|href)\s*=\s*\"(https?:)?//", APP_INDEX)
