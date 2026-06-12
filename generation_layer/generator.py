"""
generation_layer/generator.py

Research-grade answer generator.

Upgraded from V4:
  - Modality-specific citation format:
      Text:   "The torque is 45 Nm [chunk_abc]"
      Video:  "At 4:32 [seg_0012], the instructor shows..."
      Image:  "Figure 3 [img_0045] illustrates..."
      Table:  "Row 47 [tbl_0023] records..."
  - Inline UNVERIFIED: prefix for claims without evidence
  - Explicit INSUFFICIENT EVIDENCE: when evidence is empty
  - Captures raw_confidence and raw_missing from LLM output
  - Sub-questions block shown to LLM for Agent path synthesis
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models.pipeline_models import DraftAnswer

GENERATOR_SYSTEM = """\
You are a precise technical document assistant for industrial vehicle manufacturing.
You answer questions ONLY from the evidence provided.
You cite every factual claim so engineers can verify the source.
You never hallucinate — if evidence is insufficient you say so explicitly.\
"""

GENERATOR_PROMPT = """\
Answer the user question using ONLY the evidence below.

Question: {question}
Query intent: {intent_type}
Retrieval path: {retrieval_path}
{sub_questions_block}

Evidence (ordered by relevance):
{evidence_blocks}

Synthesis instruction: {synthesis_instruction}

Requirements:
1. Cite every factual claim using the EXACT chunk_id shown in the evidence header (e.g., [d2e2e965c20a_chunk_048]).
   Do NOT add any prefix — use the id exactly as it appears after "chunk_id:" in the evidence block.
   - Text/document evidence: "The value is 45 [d2e2e965c20a_chunk_003]"
   - Video segment: "At 4:32 [d2e2e965c20a_seg_012], the narrator explains..."
   - Image/diagram: "Figure 3 [d2e2e965c20a_img_045] illustrates..."
   - Table/spreadsheet: "Row 47 [d2e2e965c20a_tbl_023] shows..."
2. If a claim cannot be cited from evidence, prefix it: "UNVERIFIED: ..."
3. If evidence is completely insufficient, begin with:
   "INSUFFICIENT EVIDENCE: " then list what is missing
4. Adapt language to the domain of the evidence (academic, technical, or general).
5. End with these exact labeled fields:
   CONFIDENCE: <0.0-1.0>
   MISSING: <what would improve this answer, or "none">
   FOLLOW_UP:
   1. <follow-up question>
   2. <follow-up question>

Answer:\
"""

SUB_QUESTIONS_BLOCK = """\
Sub-questions answered (Agent path):
{sub_qs}
"""


class Generator:

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def generate(
        self,
        question:             str,
        evidence_blocks:      str,
        intent_type:          str        = "definition",
        retrieval_path:       str        = "rag",
        synthesis_instruction: str       = "",
        sub_questions:        List[Dict] = None,
    ) -> DraftAnswer:
        """
        Generate answer from evidence.

        Args:
            question:              User question
            evidence_blocks:       Formatted evidence from plugins
            intent_type:           From QueryAnalysis
            retrieval_path:        "rag" or "agent"
            synthesis_instruction: From QueryDecomposer (agent path)
            sub_questions:         Sub-questions answered (agent path, for context)

        Returns:
            DraftAnswer with text, cited_chunk_ids, raw_confidence, raw_missing
        """
        if not self.llm:
            return DraftAnswer(
                text     = "INSUFFICIENT EVIDENCE: generation backend is unavailable.",
                abstained = True,
            )

        sub_qs_block = ""
        if sub_questions and retrieval_path == "agent":
            formatted = "\n".join(
                f"  {i+1}. {sq.get('question','')}"
                for i, sq in enumerate(sub_questions)
            )
            sub_qs_block = SUB_QUESTIONS_BLOCK.format(sub_qs=formatted)

        prompt = GENERATOR_PROMPT.format(
            question             = question,
            intent_type          = intent_type,
            retrieval_path       = retrieval_path,
            sub_questions_block  = sub_qs_block,
            evidence_blocks      = evidence_blocks[:10000] if evidence_blocks else "No evidence retrieved.",
            synthesis_instruction = synthesis_instruction or "Answer directly from evidence.",
        )

        try:
            text = self.llm.invoke(
                system     = GENERATOR_SYSTEM,
                prompt     = prompt,
                max_tokens = 1500,
            )
        except Exception as e:
            logger.error(f"  Generator LLM failed: {e}")
            return DraftAnswer(
                text     = f"INSUFFICIENT EVIDENCE: generation failed — {e}",
                abstained = True,
            )

        text = text or ""

        # Extract inline citations — handles both [id] and [chunk_id: id] formats
        cited_ids = list(dict.fromkeys(re.findall(r'\[(?:chunk_id:\s*)?([A-Za-z0-9_.:/-]+)\]', text)))

        # Extract CONFIDENCE field
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        raw_conf   = float(conf_match.group(1)) if conf_match else 0.5
        raw_conf   = max(0.0, min(1.0, raw_conf))

        # Extract MISSING field
        miss_match = re.search(r"MISSING:\s*(.+?)(?=FOLLOW_UP:|$)", text, re.S)
        raw_missing = miss_match.group(1).strip() if miss_match else ""

        abstained = (
            "INSUFFICIENT EVIDENCE" in text.upper()
            and not cited_ids
        )

        logger.info(
            f"  Generator: {len(text)} chars | "
            f"citations={len(cited_ids)} | "
            f"confidence={raw_conf:.2f} | "
            f"abstained={abstained}"
        )

        return DraftAnswer(
            text            = text,
            cited_chunk_ids = cited_ids,
            abstained       = abstained,
            raw_confidence  = raw_conf,
            raw_missing     = raw_missing,
        )

    def extract_answer_body(self, text: str) -> str:
        """Strip the labeled fields from the answer body."""
        for marker in ("CONFIDENCE:", "MISSING:", "FOLLOW_UP:"):
            if marker in text:
                text = text[:text.index(marker)]
        return text.strip()

    def extract_follow_ups(self, text: str) -> List[str]:
        """Extract follow-up questions from labeled section."""
        if "FOLLOW_UP:" not in text:
            return []
        tail = text.split("FOLLOW_UP:", 1)[1]
        return [
            item.strip()
            for item in re.findall(r"\d+\.\s*(.+)", tail)
            if item.strip()
        ]