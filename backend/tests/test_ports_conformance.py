"""The current adapters satisfy the provider ports in app.ports (method names and signatures)."""

import inspect

import pytest
from app import ports
from app.agent import fallback
from app.kit.client import KitClient
from app.knowledge import rag
from app.voice.deepgram import DeepgramSTT, DeepgramTTS


def _port_methods(proto: type) -> list[str]:
    return [
        name
        for name, value in vars(proto).items()
        if inspect.isfunction(value) and (not name.startswith("_") or name == "__call__")
    ]


def _params(fn) -> list[tuple[str, object]]:
    sig = inspect.signature(fn)
    return [(p.name, p.default) for p in sig.parameters.values() if p.name != "self"]


@pytest.mark.parametrize(
    ("proto", "adapter"),
    [
        (ports.SpeechToText, DeepgramSTT),
        (ports.TextToSpeech, DeepgramTTS),
        (ports.CollegeSystem, KitClient),
    ],
)
def test_class_adapter_matches_port(proto, adapter):
    for name in _port_methods(proto):
        impl = getattr(adapter, name, None)
        assert callable(impl), f"{adapter.__name__} is missing {name}"
        assert _params(impl) == _params(getattr(proto, name)), f"{adapter.__name__}.{name} signature"


def test_fallback_module_is_a_chat_model():
    assert isinstance(fallback, ports.ChatModel)
    assert inspect.iscoroutinefunction(fallback.answer)
    assert _params(fallback.answer) == _params(ports.ChatModel.answer)


def test_embed_texts_is_an_embedder():
    assert isinstance(rag.embed_texts, ports.Embedder)
    assert inspect.iscoroutinefunction(rag.embed_texts)
    assert _params(rag.embed_texts) == _params(ports.Embedder.__call__)


def test_runtime_checkable_instances():
    assert isinstance(DeepgramSTT("key", "nova-3"), ports.SpeechToText)
    assert isinstance(KitClient(), ports.CollegeSystem)
