"""App lifespan: the prompt-cache keep-warm task (AGT5, REL1). warm_cache is always faked here,
so nothing reaches the model API."""

import asyncio

import pytest
from app.config import get_settings
from app.db import init_db

from tests.conftest import load_main


@pytest.fixture
def life(monkeypatch, fake_college):
    """Lifespan with a faked sync loop, warm_cache and first-sync signal."""
    main = load_main()
    from app.agent import loop as agent_loop

    state = {"calls": 0, "fail": 0, "hang": False, "synced": asyncio.Event()}

    async def noop():
        return None

    async def wait_first_sync():
        await state["synced"].wait()

    async def warm():
        state["calls"] += 1
        if state["fail"]:
            state["fail"] -= 1
            raise RuntimeError("model down")
        if state["hang"]:
            await asyncio.sleep(3600)

    monkeypatch.setattr(main.store, "start_sync_loop", noop)
    monkeypatch.setattr(main.store, "stop_sync_loop", noop)
    monkeypatch.setattr(main.store, "wait_first_sync", wait_first_sync, raising=False)
    monkeypatch.setattr(agent_loop, "warm_cache", warm)
    monkeypatch.setattr(main, "WARM_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(get_settings(), "llm_keep_warm", True)
    state["main"] = main
    yield state


async def until(cond, timeout=2.0):
    end = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > end:
            raise AssertionError("condition not met")
        await asyncio.sleep(0.005)


def test_agt5_keep_warm_setting_defaults_on():
    from app.config import Settings

    assert Settings.model_fields["llm_keep_warm"].default is True


async def test_agt5_warm_after_first_sync_then_periodically(life):
    main = life["main"]
    async with main.app.router.lifespan_context(main.app):
        await asyncio.sleep(0.05)
        assert life["calls"] == 0  # not before the first document sync
        life["synced"].set()
        await until(lambda: life["calls"] >= 3)
    after = life["calls"]
    await asyncio.sleep(0.05)
    assert life["calls"] == after  # cancelled on shutdown
    await init_db()


async def test_agt5_warm_task_survives_exceptions(life):
    main = life["main"]
    life["fail"] = 2
    life["synced"].set()
    async with main.app.router.lifespan_context(main.app):
        await until(lambda: life["calls"] >= 4)
    await init_db()


async def test_agt5_shutdown_cancels_a_hanging_warm_call(life):
    main = life["main"]
    life["hang"] = True
    life["synced"].set()
    async with main.app.router.lifespan_context(main.app):
        await until(lambda: life["calls"] == 1)
    await init_db()


async def test_agt5_disabled_by_setting(life, monkeypatch):
    main = life["main"]
    monkeypatch.setattr(get_settings(), "llm_keep_warm", False)
    life["synced"].set()
    async with main.app.router.lifespan_context(main.app):
        await asyncio.sleep(0.05)
    assert life["calls"] == 0
    await init_db()


async def test_agt5_first_sync_wait_is_bounded(life, monkeypatch):
    main = life["main"]
    monkeypatch.setattr(main, "WARM_FIRST_SYNC_TIMEOUT", 0.02)
    async with main.app.router.lifespan_context(main.app):
        await until(lambda: life["calls"] >= 1)  # never synced, warms anyway
    await init_db()


async def test_agt5_without_first_sync_signal_uses_start_delay(life, monkeypatch):
    main = life["main"]
    monkeypatch.delattr(main.store, "wait_first_sync")
    monkeypatch.setattr(main, "WARM_START_DELAY", 0.01)
    async with main.app.router.lifespan_context(main.app):
        await until(lambda: life["calls"] >= 1)
    await init_db()


async def test_agt5_failing_first_sync_signal_still_warms(life, monkeypatch):
    main = life["main"]

    async def broken():
        raise RuntimeError("store bug")

    monkeypatch.setattr(main.store, "wait_first_sync", broken, raising=False)
    async with main.app.router.lifespan_context(main.app):
        await until(lambda: life["calls"] >= 1)
    await init_db()


async def test_app_client_runs_with_keep_warm_off(app_client):
    assert get_settings().llm_keep_warm is False
    assert (await app_client.get("/healthz")).status_code == 200


async def test_agt5_shutdown_while_waiting_for_first_sync(life):
    main = life["main"]
    async with main.app.router.lifespan_context(main.app):
        await asyncio.sleep(0.02)  # first sync never completes
    assert life["calls"] == 0
    await init_db()
