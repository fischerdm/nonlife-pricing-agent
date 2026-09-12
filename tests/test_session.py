"""Unit tests for the pure (non-Streamlit) helpers in dashboard/_session.py.

Most of _session.py touches st.session_state/st.error and needs a real
Streamlit session to exercise meaningfully (see dashboard/glm_workbench.py's
own test file for the same convention) — but llm_available() and
demo_mode_notices() are side-effect-free by design (see their docstrings),
so they're tested directly here.
"""
import pytest

from dashboard._session import LLM_NOT_CONFIGURED_MSG, demo_mode_notices, llm_available


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


def test_demo_mode_notices_empty_when_dataset_and_key_present(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-test-key")
    dataset_path = tmp_path / "data.csv"
    dataset_path.write_text("a,b\n1,2\n")
    cfg = {"data": {"path": str(dataset_path)}}

    assert demo_mode_notices(cfg) == []


def test_demo_mode_notices_flags_missing_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-test-key")
    cfg = {"data": {"path": str(tmp_path / "does_not_exist.csv")}}

    notices = demo_mode_notices(cfg)

    assert len(notices) == 1
    assert "does_not_exist.csv" in notices[0]


def test_demo_mode_notices_flags_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    dataset_path = tmp_path / "data.csv"
    dataset_path.write_text("a,b\n1,2\n")
    cfg = {"data": {"path": str(dataset_path)}}

    assert demo_mode_notices(cfg) == [LLM_NOT_CONFIGURED_MSG]


def test_demo_mode_notices_flags_both(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = {"data": {"path": str(tmp_path / "does_not_exist.csv")}}

    notices = demo_mode_notices(cfg)

    assert len(notices) == 2
    assert "does_not_exist.csv" in notices[0]
    assert notices[1] == LLM_NOT_CONFIGURED_MSG
