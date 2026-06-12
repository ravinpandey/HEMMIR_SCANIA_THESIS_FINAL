"""
enrichment_layer/enrichers/_parse_utils.py

Shared parse helpers used by every enricher.
No Pydantic here — enrichers work on plain dicts.
Validation happens at the disk→memory boundary in shared/models/metadata_models.py.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


def extract_field(text: str, key: str) -> Optional[str]:
    """
    Extract a labeled field from LLM output.

    Handles both KEY: value and KEY:\nvalue patterns.
    Returns None if field is missing or empty.

    Example:
        text = "SUMMARY: Covers pump maintenance.\\nCONFIDENCE: 0.9"
        extract_field(text, "SUMMARY") → "Covers pump maintenance."
    """
    m = re.search(
        rf"^{re.escape(key)}:\s*(.+?)(?=\n[A-Z_]{{2,}}:|$)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not m:
        return None
    val = m.group(1).strip()
    return val if val else None


def extract_float(
    text: str, key: str, default: Optional[float] = None
) -> Optional[float]:
    """
    Extract a float confidence score from a labeled field.
    Clamps to [0.0, 1.0]. Returns default if not found.
    """
    m = re.search(rf"^{re.escape(key)}:\s*([\d.]+)", text, re.MULTILINE)
    if not m:
        return default
    try:
        return round(max(0.0, min(1.0, float(m.group(1)))), 4)
    except ValueError:
        return default


def extract_list(text: str, key: str) -> List[str]:
    """
    Extract a comma-separated list from a labeled field.
    Returns [] if field is missing or value is "none" / "n/a".
    """
    raw = extract_field(text, key)
    if not raw or raw.lower().strip() in ("none", "n/a", "-", ""):
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def is_complete(text: str, required_keys: List[str]) -> bool:
    """Return True only if all required keys are present and non-empty."""
    for key in required_keys:
        if not extract_field(text, key):
            return False
    return True


def missing_keys(text: str, required_keys: List[str]) -> List[str]:
    """Return list of required keys that are absent or empty in the response."""
    return [k for k in required_keys if not extract_field(text, k)]


def invoke_with_retry(
    llm_client,
    first_prompt: str,
    retry_prompt_template: str,
    required_keys: List[str],
    *,
    system: str = "",
    use_cache: bool = False,
    system_doc: str = "",
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> Optional[str]:
    """
    Call the LLM with up to max_retries retries on incomplete parse.

    Args:
        llm_client: AnthropicClient or BedrockClient instance
        first_prompt: initial user prompt
        retry_prompt_template: template with {missing} and {prev_response} placeholders
        required_keys: keys that must be present in the response
        system: plain system prompt (used when use_cache=False)
        use_cache: whether to use invoke_with_cache
        system_doc: document text for caching (used when use_cache=True)
        max_retries: number of retry attempts after the first call
        retry_delay: seconds between retries

    Returns:
        Last response text, or None if all attempts failed.
    """
    import time

    prompt   = first_prompt
    response = None

    for attempt in range(1, max_retries + 2):
        try:
            if use_cache:
                response = llm_client.invoke_with_cache(
                    system_doc=system_doc,
                    prompt=prompt,
                )
            else:
                response = llm_client.invoke(
                    system=system,
                    prompt=prompt,
                )
        except Exception as e:
            from loguru import logger
            logger.warning(f"  LLM call failed (attempt {attempt}): {e}")
            if attempt <= max_retries:
                time.sleep(retry_delay * attempt)
            continue

        missing = missing_keys(response, required_keys)
        if not missing:
            return response

        from loguru import logger
        logger.debug(f"  attempt {attempt}: missing keys {missing}")

        if attempt <= max_retries:
            prompt = retry_prompt_template.format(
                missing=", ".join(missing),
                prev_response=response[:600],
            )
            time.sleep(retry_delay)

    return response
