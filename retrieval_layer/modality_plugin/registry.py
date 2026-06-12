"""
retrieval_layer/modality_plugin/registry.py

Plugin registry — builds all 8 modality plugins and returns them
as a dict keyed by modality_name.

8 plugins total:
  Existing (from V4, unchanged):
    text_to_text     — text query → text chunks (1536-dim)
    image_to_image   — image query → image chunks (CLIP 512-dim)
    table_to_table   — table/text query → table chunks (1536-dim)
    text_to_image    — text query → images via CLIP (cross-modal)
    text_to_table    — text query → tables via text (cross-modal)
    image_to_text    — image query → text chunks (cross-modal)

  New (HEMMIR video support):
    video_segment    — text query → video segments (1536-dim ASR)
    video_frame      — CLIP query → video frames (512-dim) + text

The registry is used by:
  - RAG path: retrieve_chunks.py dispatches to plugins by modality name
  - Agent path: iterative_retriever uses plugin names for collection routing
  - Generation layer: format_prompt_block per evidence item modality
"""

from __future__ import annotations

from retrieval_layer.modality_plugin.base import ModalityPlugin
from retrieval_layer.modality_plugin.cross_modality.image_to_text import ImageToTextPlugin
from retrieval_layer.modality_plugin.cross_modality.text_to_image import TextToImagePlugin
from retrieval_layer.modality_plugin.cross_modality.text_to_table import TextToTablePlugin
from retrieval_layer.modality_plugin.same_modality.image_to_image import ImageToImagePlugin
from retrieval_layer.modality_plugin.same_modality.table_to_table import TableToTablePlugin
from retrieval_layer.modality_plugin.same_modality.text_to_text import TextToTextPlugin
from retrieval_layer.modality_plugin.video.video_segment_plugin import VideoSegmentPlugin
from retrieval_layer.modality_plugin.video.video_frame_plugin import VideoFramePlugin


def build_plugin_registry(
    text_embedder=None,
    image_embedder=None,
    llm_client=None,
    store=None,
) -> dict[str, ModalityPlugin]:
    """
    Build and return all modality plugins.

    Args:
        text_embedder:  TextEmbedder instance (1536-dim OpenAI)
        image_embedder: ImageEmbedder instance (CLIP 512-dim)
        llm_client:     LLMClient instance (for cross-modal description)

    Returns:
        Dict[modality_name → plugin instance]
    """
    plugins = [
        # ── Existing V4 plugins ────────────────────────────────────────
        TextToTextPlugin(
            text_embedder  = text_embedder,
            image_embedder = image_embedder,
            bedrock_client = llm_client,
            store          = store,
        ),
        ImageToImagePlugin(
            text_embedder  = text_embedder,
            image_embedder = image_embedder,
            bedrock_client = llm_client,
            store          = store,
        ),
        TableToTablePlugin(
            text_embedder  = text_embedder,
            image_embedder = image_embedder,
            bedrock_client = llm_client,
            store          = store,
        ),
        TextToImagePlugin(
            text_embedder  = text_embedder,
            image_embedder = image_embedder,
            bedrock_client = llm_client,
            store          = store,
        ),
        TextToTablePlugin(
            text_embedder  = text_embedder,
            image_embedder = image_embedder,
            bedrock_client = llm_client,
            store          = store,
        ),
        ImageToTextPlugin(
            text_embedder  = text_embedder,
            image_embedder = image_embedder,
            bedrock_client = llm_client,
            store          = store,
        ),

        # ── New video plugins ──────────────────────────────────────────
        VideoSegmentPlugin(
            text_embedder  = text_embedder,
            image_embedder = image_embedder,
            bedrock_client = llm_client,
            store          = store,
        ),
        VideoFramePlugin(
            text_embedder  = text_embedder,
            image_embedder = image_embedder,
            bedrock_client = llm_client,
            store          = store,
        ),
    ]

    return {plugin.modality_name: plugin for plugin in plugins}