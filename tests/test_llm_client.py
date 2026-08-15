from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from core.llm_client import LLMClient


class _Echo(BaseModel):
    value: str


def _response(text: str) -> MagicMock:
    r = MagicMock()
    r.content = [MagicMock(text=text)]
    return r


@pytest.fixture
def client():
    c = LLMClient(api_key="test-key")
    c.client = MagicMock()
    return c


def test_call_retries_once_on_malformed_json_then_succeeds(client):
    client.client.messages.create.side_effect = [
        _response('{"value": "a" "b"}'),  # missing comma — malformed
        _response('{"value": "ok"}'),
    ]

    result = client.call("prompt", _Echo)

    assert result.value == "ok"
    assert client.client.messages.create.call_count == 2


def test_call_raises_last_error_after_exhausting_retries(client):
    client.client.messages.create.return_value = _response('{"value": "a" "b"}')

    with pytest.raises(Exception):
        client.call("prompt", _Echo)

    assert client.client.messages.create.call_count == 3  # _MAX_ATTEMPTS


def test_call_succeeds_first_try_without_retry(client):
    client.client.messages.create.return_value = _response('{"value": "ok"}')

    result = client.call("prompt", _Echo)

    assert result.value == "ok"
    assert client.client.messages.create.call_count == 1
