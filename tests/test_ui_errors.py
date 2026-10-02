import logging

import groq
import httpx
import pytest
from cohere.core.api_error import ApiError as CohereApiError

from policyguard.ui.errors import (
    RATE_LIMITED,
    SERVICE_UNAVAILABLE,
    UNEXPECTED,
    friendly_error_message,
)

_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _groq_status_error(cls, status_code: int):
    return cls("raw provider error", response=httpx.Response(status_code, request=_REQUEST), body=None)


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_groq_status_error(groq.RateLimitError, 429), RATE_LIMITED),
        (CohereApiError(status_code=429, body="too many requests"), RATE_LIMITED),
        # The 403 "Access denied. Please check your network settings." seen on Streamlit Cloud.
        (_groq_status_error(groq.PermissionDeniedError, 403), SERVICE_UNAVAILABLE),
        (_groq_status_error(groq.InternalServerError, 503), SERVICE_UNAVAILABLE),
        (groq.APIConnectionError(request=_REQUEST), SERVICE_UNAVAILABLE),
        (groq.APITimeoutError(request=_REQUEST), SERVICE_UNAVAILABLE),
        (CohereApiError(status_code=500, body="server error"), SERVICE_UNAVAILABLE),
        (ValueError("bug in our code"), UNEXPECTED),
    ],
)
def test_friendly_error_message_maps_exception(exc, expected):
    assert friendly_error_message(exc) == expected


def test_friendly_error_message_hides_raw_error_but_logs_it(caplog):
    exc = _groq_status_error(groq.PermissionDeniedError, 403)

    with caplog.at_level(logging.ERROR, logger="policyguard.ui.errors"):
        message = friendly_error_message(exc)

    assert "raw provider error" not in message
    assert caplog.records[0].exc_info[1] is exc
