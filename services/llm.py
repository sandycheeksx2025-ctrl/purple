"""
LLM client for OpenRouter API.

Provides async interface for text generation with tool calling support.
Unified client for all LLM interactions in the bot.
"""

import json
import logging
from typing import Any

import httpx

from config.models import LLM_MODEL
from utils.api import OPENROUTER_URL, get_openrouter_headers

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for OpenRouter LLM API."""

    def __init__(self, model: str = LLM_MODEL):
        self.model = model

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _safe_json_parse(self, text: str) -> Any:
        """
        Safely attempt to parse JSON from model output.

        Returns parsed object or None if parsing fails.
        """
        try:
            return json.loads(text)
        except Exception:
            return None

    def _normalize_structured_response(
        self,
        raw_content: str,
        response_format: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Normalize LLM output into a dict matching expected structured format.

        If JSON parsing fails, fallback to wrapping text.
        """
        parsed = self._safe_json_parse(raw_content)

        if isinstance(parsed, dict):
            return parsed

        logger.warning("[LLM] Structured output invalid JSON — falling back to text wrapper")

        # Best-effort fallback depending on schema intent
        # Common patterns in your system
        if "plan" in json.dumps(response_format):
            return {
                "reasoning": raw_content.strip(),
                "plan": []
            }

        if "post_text" in json.dumps(response_format) or "post" in json.dumps(response_format):
            return {
                "post_text": raw_content.strip()
            }

        if "thinking" in json.dumps(response_format):
            return {
                "thinking": raw_content.strip()
            }

        # Generic fallback
        return {"content": raw_content.strip()}

    # ---------------------------
    # Public API
    # ---------------------------

    async def generate(self, system: str, user: str) -> str:
        """
        Simple text generation (no schema).
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers=get_openrouter_headers(),
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            logger.info(f"[LLM] Generated response: {content[:100]}...")
            return content

    async def generate_structured(
        self,
        system: str,
        user: str,
        response_format: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate structured JSON output with hard safety.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers=get_openrouter_headers(),
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 500,
                    "response_format": response_format
                }
            )
            response.raise_for_status()
            data = response.json()

            raw = data["choices"][0]["message"]["content"]
            logger.info(f"[LLM] Generated structured response (raw): {raw[:200]}")

            return self._normalize_structured_response(raw, response_format)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Multi-turn chat completion with schema tolerance.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024
        }

        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers=get_openrouter_headers(),
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            raw = data["choices"][0]["message"]["content"]
            logger.info(f"[LLM] Chat response (raw): {raw[:200]}...")

            if response_format:
                return self._normalize_structured_response(raw, response_format)

            return {"content": raw}
