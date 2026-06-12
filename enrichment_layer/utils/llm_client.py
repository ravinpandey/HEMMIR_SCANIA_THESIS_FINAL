"""
enrichment_layer/utils/llm_client.py

Abstract LLM client with two concrete backends:
    AnthropicClient  — uses Anthropic Python SDK directly (ANTHROPIC_API_KEY)
    BedrockClient    — uses boto3 (AWS credentials via env / IAM role)

Both implement the same three methods:
    invoke(system, prompt, max_tokens)            — single call, no cache
    invoke_with_cache(system_doc, prompt, max_tokens) — system prompt cached
    invoke_with_image(prompt, image_b64, media_type, system, max_tokens) — vision

Prompt caching strategy:
    The full document text is sent in the system prompt with cache_control=ephemeral.
    All per-chunk calls reuse that cache — only the small user prompt is charged.
    On Anthropic API: cache is valid for 5 minutes (refreshed on each read).
    On Bedrock: cache behaviour follows the model's cache_control support.

Models used:
    Enrichment text    : claude-haiku-4-5  (fast, cheap, good for structured output)
    Vision / summaries : claude-sonnet-4-5 (better multimodal + long-form reasoning)
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Optional
from loguru import logger


# ── Model IDs ─────────────────────────────────────────────────────────────────
ANTHROPIC_HAIKU_MODEL  = "claude-haiku-4-5-20251001"
ANTHROPIC_SONNET_MODEL = "claude-sonnet-4-5-20251001"

BEDROCK_HAIKU_MODEL  = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_SONNET_MODEL = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"

CACHE_SUPPORTED = {
    ANTHROPIC_HAIKU_MODEL,
    ANTHROPIC_SONNET_MODEL,
    BEDROCK_HAIKU_MODEL,
    BEDROCK_SONNET_MODEL,
}



# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════════

class LLMClient(ABC):

    @abstractmethod
    def invoke(
        self,
        system:      str,
        prompt:      str,
        max_tokens:  int            = 500,
        temperature: Optional[float] = None,
    ) -> str:
        """Single call — no prompt caching."""

    @abstractmethod
    def invoke_with_cache(
        self,
        system_doc: str,
        prompt:     str,
        max_tokens: int = 500,
    ) -> str:
        """
        Call with system prompt cached.
        system_doc: full document text — cached after first call per document.
        prompt: per-chunk user prompt — small, charged every call.
        """

    @abstractmethod
    def invoke_with_image(
        self,
        prompt:     str,
        image_b64:  str,
        media_type: str,
        system:     str  = "",
        max_tokens: int  = 600,
    ) -> str:
        """Vision call. image_b64: base64-encoded image bytes."""


# ═══════════════════════════════════════════════════════════════════════════════
# Anthropic SDK backend
# ═══════════════════════════════════════════════════════════════════════════════

class AnthropicClient(LLMClient):
    """
    Uses the official Anthropic Python SDK.
    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(
        self,
        text_model:   str = ANTHROPIC_HAIKU_MODEL,
        vision_model: str = ANTHROPIC_SONNET_MODEL,
        max_retries:  int = 3,
    ):
        try:
            import anthropic
            self._client      = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
            self._anthropic   = anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self.text_model   = text_model
        self.vision_model = vision_model
        self.max_retries  = max_retries
        logger.info(f"AnthropicClient ready | text={text_model} | vision={vision_model}")

    def invoke(
        self,
        system:      str,
        prompt:      str,
        max_tokens:  int            = 500,
        temperature: Optional[float] = None,
    ) -> str:
        return self._call(
            model=self.text_model,
            system=[{"type": "text", "text": system}],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def invoke_with_cache(
        self, system_doc: str, prompt: str, max_tokens: int = 500
    ) -> str:
        system = [
            {
                "type": "text",
                "text": system_doc,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages = [{"role": "user", "content": prompt}]
        return self._call(
            model=self.text_model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )

    def invoke_with_image(
        self,
        prompt:     str,
        image_b64:  str,
        media_type: str,
        system:     str = "",
        max_tokens: int = 600,
    ) -> str:
        content = [
            {
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": media_type,
                    "data":       image_b64,
                },
            },
            {"type": "text", "text": prompt},
        ]
        sys_block = (
            [{"type": "text", "text": system}] if system else []
        )
        return self._call(
            model=self.vision_model,
            system=sys_block,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )

    def _call(
        self,
        model:       str,
        system:      list,
        messages:    list,
        max_tokens:  int,
        temperature: Optional[float] = None,
    ) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs = dict(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                if system:
                    kwargs["system"] = system
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = self._client.messages.create(**kwargs)
                usage = getattr(response, "usage", None)
                if usage:
                    logger.debug(
                        f"  tokens in={usage.input_tokens} "
                        f"out={usage.output_tokens} "
                        f"cache_read={getattr(usage,'cache_read_input_tokens',0)} "
                        f"cache_write={getattr(usage,'cache_creation_input_tokens',0)}"
                    )
                return response.content[0].text
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"  AnthropicClient attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    raise
                time.sleep(wait)
        raise RuntimeError("AnthropicClient: max retries exceeded")


# ═══════════════════════════════════════════════════════════════════════════════
# AWS Bedrock backend
# ═══════════════════════════════════════════════════════════════════════════════

class BedrockClient(LLMClient):
    """
    Uses boto3 Bedrock runtime. Credential resolution: env vars → IAM role.
    """

    def __init__(
        self,
        region:       str = "eu-west-1",
        text_model:   str = BEDROCK_HAIKU_MODEL,
        vision_model: str = BEDROCK_SONNET_MODEL,
        max_retries:  int = 3,
    ):
        try:
            import boto3, json as _json
            self._boto3  = boto3
            self._json   = _json
            self._client = boto3.client("bedrock-runtime", region_name=region)
        except ImportError:
            raise ImportError("pip install boto3")
        self.text_model   = text_model
        self.vision_model = vision_model
        self.max_retries  = max_retries
        logger.info(f"BedrockClient ready | region={region} | text={text_model}")

    def invoke(
        self,
        system:      str,
        prompt:      str,
        max_tokens:  int            = 500,
        temperature: Optional[float] = None,
    ) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        return self._call(self.text_model, body)

    def invoke_with_cache(
        self, system_doc: str, prompt: str, max_tokens: int = 500
    ) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_doc,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": prompt}],
        }
        return self._call(self.text_model, body)

    def invoke_with_image(
        self,
        prompt:     str,
        image_b64:  str,
        media_type: str,
        system:     str = "",
        max_tokens: int = 600,
    ) -> str:
        content = [
            {
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": media_type,
                    "data":       image_b64,
                },
            },
            {"type": "text", "text": prompt},
        ]
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            body["system"] = system
        return self._call(self.vision_model, body)

    def _call(self, model_id: str, body: dict) -> str:
        import json
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(response["body"].read())
                content = result.get("content", [])
                if not content:
                    raise ValueError(f"Bedrock returned empty content (stop_reason={result.get('stop_reason')})")
                return content[0]["text"]
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"  BedrockClient attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    raise
                time.sleep(wait)
        raise RuntimeError("BedrockClient: max retries exceeded")


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_llm_client(provider: str = "anthropic", **kwargs) -> LLMClient:
    """
    Factory function used by enrichment_pipeline.py.

    Args:
        provider: "anthropic" or "bedrock"
        **kwargs: passed to the client constructor
                  (region= for bedrock, text_model= / vision_model= for both)
    """
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "bedrock":
        return BedrockClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose 'anthropic' or 'bedrock'.")
