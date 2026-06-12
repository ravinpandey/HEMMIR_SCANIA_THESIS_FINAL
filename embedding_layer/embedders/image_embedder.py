"""
embedding_layer/embedders/image_embedder.py

Dual embedding for image chunks and video frames.

CLIP embedding  (512-dim)  ← PIL image → visual similarity / cross-modal queries
OpenAI text     (1536-dim) ← caption + summary + component + annotations → text queries

Both vectors stored on every chunk:
  clip_embedding  : image_chunks_clip collection  (512-dim cosine)
  text_embedding  : image_chunks_text collection  (1536-dim cosine)

Video frames go through the same path — frame_role and parent transcript
context are already baked into contextual_summary by video_frame_enricher.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from shared.models.metadata_models import CLIP_EMBED_DIM, TEXT_EMBED_DIM


def _build_image_text(chunk: Dict[str, Any]) -> str:
    parts = [
        chunk.get("image_caption")       or "",
        chunk.get("contextual_summary")  or "",
        chunk.get("depicted_component")  or "",
        chunk.get("visible_annotations") or "",
        chunk.get("image_type")          or "",
        chunk.get("frame_role")          or "",
    ]
    return " ".join(p.strip() for p in parts if p.strip())


def _is_embeddable(text: str) -> bool:
    return bool(text and len(text.strip()) >= 5)


class ImageEmbedder:

    def __init__(
        self,
        clip_model:      str = "ViT-B-32",
        clip_pretrained: str = "openai",
        text_model:      str = "text-embedding-3-small",
    ):
        # ── Load CLIP ──────────────────────────────────────────────────
        logger.info(f"Loading CLIP: {clip_model} ({clip_pretrained})")
        try:
            import open_clip
            import torch
            self._clip_model, _, self._preprocess = \
                open_clip.create_model_and_transforms(
                    clip_model, pretrained=clip_pretrained
                )
            self._clip_model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model.to(self._device)
            self._torch = torch
            logger.success(f"CLIP loaded on {self._device}")
        except ImportError:
            raise ImportError("pip install open-clip-torch torch")

        # ── Load OpenAI text embedder ──────────────────────────────────
        logger.info(f"Loading OpenAI text embedder for images: {text_model}")
        try:
            from openai import OpenAI
            self._openai  = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            self._text_model = text_model
        except ImportError:
            raise ImportError("pip install openai")
        logger.success(f"OpenAI image text embedder ready ({TEXT_EMBED_DIM}-dim)")

    def embed_chunks(
        self,
        chunks:          List[Dict[str, Any]],
        doc_output_dir:  Path,
        image_batch_size: int = 8,
        text_batch_size:  int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Embed image chunks with CLIP (visual) + OpenAI text (caption/summary).
        Idempotent: skips chunks that already have clip_embedding.
        """
        to_embed = [(i, c) for i, c in enumerate(chunks) if not c.get("clip_embedding")]
        if not to_embed:
            logger.info("  ImageEmbedder: all chunks already embedded")
            return chunks

        logger.info(f"  ImageEmbedder: embedding {len(to_embed)}/{len(chunks)} images")

        chunks = self._embed_clip(chunks, to_embed, doc_output_dir, image_batch_size)
        chunks = self._embed_text(chunks, to_embed, text_batch_size)
        return chunks

    # ── CLIP visual embeddings ─────────────────────────────────────────

    def _embed_clip(
        self,
        chunks:         List[Dict],
        to_embed:       List[Tuple[int, Dict]],
        doc_output_dir: Path,
        batch_size:     int,
    ) -> List[Dict]:
        from PIL import Image as PILImage

        images: List[Any] = []
        valid_indices: List[int] = []

        for i, chunk in to_embed:
            img_path = self._resolve_path(doc_output_dir, chunk)
            if not img_path:
                continue
            try:
                img = PILImage.open(img_path).convert("RGB")
                images.append(self._preprocess(img))
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"  Cannot load image {chunk.get('chunk_id')}: {e}")

        if not images:
            return chunks

        for start in tqdm(range(0, len(images), batch_size), desc="CLIP embed"):
            batch   = images[start : start + batch_size]
            v_idxs  = valid_indices[start : start + batch_size]
            tensor  = self._torch.stack(batch).to(self._device)
            with self._torch.no_grad():
                feats = self._clip_model.encode_image(tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            for idx, emb in zip(v_idxs, feats.cpu().tolist()):
                chunks[idx]["clip_embedding"] = emb

        logger.success(f"  CLIP: {len(valid_indices)} × {CLIP_EMBED_DIM}-dim")
        return chunks

    # ── OpenAI text embeddings for images ─────────────────────────────

    def _embed_text(
        self,
        chunks:    List[Dict],
        to_embed:  List[Tuple[int, Dict]],
        batch_size: int,
    ) -> List[Dict]:
        pairs = [
            (i, _build_image_text(chunks[i]))
            for i, _ in to_embed
        ]
        valid = [(i, t) for i, t in pairs if _is_embeddable(t)]
        if not valid:
            return chunks

        indices = [i for i, _ in valid]
        texts   = [t for _, t in valid]
        all_embs: List[List[float]] = []

        for start in tqdm(range(0, len(texts), batch_size), desc="OpenAI(img) embed"):
            batch = texts[start : start + batch_size]
            for attempt in range(3):
                try:
                    resp = self._openai.embeddings.create(
                        input=batch, model=self._text_model
                    )
                    all_embs.extend([r.embedding for r in resp.data])
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"  OpenAI retry {attempt+1}/3: {e}")
                    time.sleep(2 ** attempt)

        for idx, emb in zip(indices, all_embs):
            chunks[idx]["text_embedding"] = emb

        logger.success(f"  OpenAI(img): {len(all_embs)} × {TEXT_EMBED_DIM}-dim")
        return chunks

    # ── Path helper ────────────────────────────────────────────────────

    def _resolve_path(self, doc_output_dir: Path, chunk: Dict) -> Optional[Path]:
        rel = chunk.get("image_path", "")
        if not rel:
            return None
        p = doc_output_dir / rel
        return p if p.exists() else None
