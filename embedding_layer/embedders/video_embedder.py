"""
embedding_layer/embedders/video_embedder.py

Embedding for video-specific chunk types.

VideoSegmentChunk  → text_embedder.embed_video_segments()
    Routed through _encode_text() in encoding layer.
    Input: contextual_summary + "\\n\\n" + transcript_text
    Output: text_embedding (1536-dim) → video_segments collection

VideoFrameChunk    → image_embedder.embed_chunks()
    Routed through _encode_image() in encoding layer.
    Input: image file (CLIP) + image_caption + contextual_summary (OpenAI)
    Output:
      clip_embedding (512-dim)  → video_frames_clip collection
      text_embedding (1536-dim) → image_chunks_text collection
                                  (with source_modality=video_frame)

This is a thin orchestration class — the heavy lifting is in TextEmbedder
and ImageEmbedder which are reused directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from embedding_layer.embedders.text_embedder  import TextEmbedder
from embedding_layer.embedders.image_embedder import ImageEmbedder


class VideoEmbedder:

    def __init__(
        self,
        text_embedder:  TextEmbedder,
        image_embedder: ImageEmbedder,
    ):
        self.text_embedder  = text_embedder
        self.image_embedder = image_embedder

    def embed_segments(
        self,
        segments:   List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Embed video segments via text pipeline.
        Input: contextual_summary + transcript_text (after enrichment).
        """
        if not segments:
            return segments
        to_embed = [s for s in segments if not s.get("text_embedding")]
        if not to_embed:
            logger.info("  VideoEmbedder: all segments already embedded")
            return segments

        logger.info(
            f"  VideoEmbedder: embedding {len(to_embed)}/{len(segments)} segments"
        )
        return self.text_embedder.embed_video_segments(segments, batch_size=batch_size)

    def embed_frames(
        self,
        frames:          List[Dict[str, Any]],
        doc_output_dir:  Path,
        image_batch_size: int = 8,
        text_batch_size:  int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Embed video frames via dual CLIP + text pipeline.
        Reuses ImageEmbedder — frame image_path resolved same as regular images.
        """
        if not frames:
            return frames
        to_embed = [f for f in frames if not f.get("clip_embedding")]
        if not to_embed:
            logger.info("  VideoEmbedder: all frames already embedded")
            return frames

        logger.info(
            f"  VideoEmbedder: embedding {len(to_embed)}/{len(frames)} frames"
        )
        return self.image_embedder.embed_chunks(
            frames, doc_output_dir,
            image_batch_size=image_batch_size,
            text_batch_size=text_batch_size,
        )
