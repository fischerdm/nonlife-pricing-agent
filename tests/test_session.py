"""Unit tests for the pure (non-Streamlit) helpers in dashboard/_session.py.

Most of _session.py touches st.session_state/st.error and needs a real
Streamlit session to exercise meaningfully (see dashboard/glm_workbench.py's
own test file for the same convention) — but llm_available() is
side-effect-free by design (see its docstring), so it's tested directly here.
"""
import pytest

from dashboard._session import llm_available


@pytest.fixture(autouse=True)
def no_real_dotenv(monkeypatch):
    # load_dotenv() searches upward from _session.py's own file location, not
    # the test's cwd, so it finds this repo's real (gitignored) .env
    # regardless of monkeypatch.chdir — and since python-dotenv doesn't
    # override an already-set var, calling it after a monkeypatch.delenv
    # would silently re-populate ANTHROPIC_API_KEY from that real file.
    # No-op it so these tests only ever see the env vars they set explicitly.
    monkeypatch.setattr("dashboard._session.load_dotenv", lambda *a, **k: None)


def test_llm_available_false_when_api_key_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm_available() is False


def test_llm_available_true_when_api_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-test-key")
    assert llm_available() is True
