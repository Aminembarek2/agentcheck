"""Generation lookup retries without network access or model calls."""

import io
import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from scripts.check_route import GENERATION_URL, lookup_provider


def test_generation_lookup_retries_until_record_is_available():
    missing = HTTPError(GENERATION_URL, 404, "Not Found", None, None)
    response = io.BytesIO(json.dumps({
        "data": {"provider_name": "Parasail"},
    }).encode())
    with (patch("urllib.request.urlopen", side_effect=[missing, response]) as get,
          patch("scripts.check_route.time.sleep") as sleep):
        assert lookup_provider("gen-test", "test-key") == ("Parasail", "")
    assert get.call_count == 2
    sleep.assert_called_once()


@pytest.mark.parametrize("status, attempts", [(401, 1), (404, 4)])
def test_generation_lookup_stops_after_bounded_retries(status, attempts):
    error = HTTPError(GENERATION_URL, status, "Unavailable", None, None)
    with (patch("urllib.request.urlopen", side_effect=error) as get,
          patch("scripts.check_route.time.sleep") as sleep):
        provider, reason = lookup_provider("gen-test", "test-key")
    assert provider == ""
    assert reason == f"HTTP {status} from {GENERATION_URL}"
    assert get.call_count == attempts
    assert sleep.call_count == attempts - 1
