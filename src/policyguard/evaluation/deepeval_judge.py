"""Custom DeepEval judge LLM backed by Groq.

DeepEval's metrics default to an OpenAI judge; this swaps in a Groq-hosted model instead. The
default judge (openai/gpt-oss-120b) is deliberately a different, larger model than the default
generator (openai/gpt-oss-20b, see policyguard.generation.chain), so the judge isn't grading its
own answers.

DeepEval passes a Pydantic `schema` for every structured step of a metric (verdicts, reasons,
...). Those calls use Groq's JSON-schema structured output and return a parsed schema instance,
which DeepEval accepts directly instead of scraping JSON out of free text.
"""

from __future__ import annotations

import os

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"

# Every metric makes several judge calls per test case, which can trip Groq's per-minute rate
# limits; the Groq SDK retries 429s/5xx with exponential backoff (honoring `retry-after`).
DEFAULT_MAX_RETRIES = 6


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model: str | None = None, max_retries: int = DEFAULT_MAX_RETRIES):
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set. Add it to a .env file or export it before running.")
        self._max_retries = max_retries
        super().__init__(model or os.environ.get("DEEPEVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))

    def load_model(self):
        from groq import AsyncGroq, Groq

        self._async_client = AsyncGroq(max_retries=self._max_retries)
        return Groq(max_retries=self._max_retries)

    def get_model_name(self) -> str:
        return self.name

    def _request(self, prompt: str, schema: type[BaseModel] | None) -> dict:
        request = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        return request

    @staticmethod
    def _parse(content: str | None, schema: type[BaseModel] | None) -> str | BaseModel:
        content = content or ""
        return schema.model_validate_json(content) if schema is not None else content

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str | BaseModel:
        response = self.model.chat.completions.create(**self._request(prompt, schema))
        return self._parse(response.choices[0].message.content, schema)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str | BaseModel:
        response = await self._async_client.chat.completions.create(**self._request(prompt, schema))
        return self._parse(response.choices[0].message.content, schema)
