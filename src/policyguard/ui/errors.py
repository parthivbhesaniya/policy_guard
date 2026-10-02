"""Turn pipeline exceptions into messages fit to show a demo visitor.

Raw provider errors (e.g. Groq's `Error code: 403 - {'error': {'message': 'Access denied...'}}`)
mean nothing to an end user and leak internals. The full exception is still logged here, and
LangSmith records it on the run's trace when tracing is on.
"""

from __future__ import annotations

import logging

import groq
from cohere.core.api_error import ApiError as CohereApiError

logger = logging.getLogger(__name__)

RATE_LIMITED = (
    "PolicyGuard is getting more questions than its AI service allows right now. "
    "Please wait a minute and try again."
)
SERVICE_UNAVAILABLE = (
    "The AI service PolicyGuard relies on is temporarily unavailable. "
    "Please try again in a minute."
)
UNEXPECTED = "Something went wrong while answering your question. Please try again."


def friendly_error_message(exc: BaseException) -> str:
    """Log `exc` with its traceback and return a user-facing message for it."""
    logger.error("PolicyGuard pipeline failed", exc_info=exc)

    if isinstance(exc, groq.RateLimitError) or (
        isinstance(exc, CohereApiError) and exc.status_code == 429
    ):
        return RATE_LIMITED
    if isinstance(exc, (groq.APIError, CohereApiError)):
        return SERVICE_UNAVAILABLE
    return UNEXPECTED
