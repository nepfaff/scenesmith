"""Turn-trimming session wrapper for efficient agent memory management.

This module provides a session wrapper that keeps only the last N turns fully
intact (with images) while trimming older turns by removing images and
optionally summarizing the text content.
"""

import hashlib
import json
import logging
import sqlite3
import tempfile
import threading

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents import SQLiteSession
from agents.items import TResponseInputItem
from omegaconf import DictConfig
from openai import AsyncOpenAI, Timeout
from openai.types.shared import Reasoning

from scenesmith.prompts import prompt_registry
from scenesmith.prompts.registry import SessionMemoryPrompts

console_logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """Represents a conversation turn starting with a user message."""

    start_index: int
    end_index: int  # exclusive
    items: list[TResponseInputItem]


def _is_user_message(item: TResponseInputItem) -> bool:
    """Check if item starts a new turn (user message)."""
    return item.get("role") == "user"


def _is_system_message(item: TResponseInputItem) -> bool:
    """Check if item is a system message (never trimmed)."""
    return item.get("role") == "system"


def _parse_turns(items: list[TResponseInputItem]) -> tuple[list[Turn], int]:
    """Parse items into turns based on user message boundaries.

    Args:
        items: List of session items.

    Returns:
        Tuple of (list of Turn objects, index where first turn starts).
        Items before the first user message are considered "preamble" and
        should be preserved intact.
    """
    turns: list[Turn] = []
    current_start: int | None = None
    first_turn_start = len(items)  # Default: no turns

    for i, item in enumerate(items):
        if not _is_user_message(item):
            continue
        if current_start is not None:
            turn = Turn(
                start_index=current_start, end_index=i, items=items[current_start:i]
            )
            turns.append(turn)
        else:
            first_turn_start = i
        current_start = i

    # Final turn (from last user message to end).
    if current_start is not None:
        turn = Turn(
            start_index=current_start, end_index=len(items), items=items[current_start:]
        )
        turns.append(turn)

    return turns, first_turn_start


def _is_image_content(part: dict[str, Any]) -> bool:
    """Detect image content in various formats.

    Handles:
    - type: "input_image" with image_url
    - type: "image_url" with url field
    - type: "image" with various formats
    - Nested image_url objects
    """
    type_ = part.get("type", "")

    # Direct image types.
    if type_ in ("input_image", "image_url", "image"):
        return True

    # Check for image_url field with data URL or HTTP URL.
    if "image_url" in part:
        url = part["image_url"]
        if isinstance(url, str):
            return url.startswith("data:image") or url.startswith("http")
        if isinstance(url, dict) and "url" in url:
            url_value = url["url"]
            if isinstance(url_value, str):
                return url_value.startswith("data:image") or url_value.startswith(
                    "http"
                )
            return True

    return False


def _contains_base64_image(text: str) -> bool:
    """Check if text contains base64-encoded image data.

    Looks for the standard data URL pattern used by ToolOutputImage:
    data:image/png;base64,{base64_data}
    """
    return "data:image/" in text and "base64," in text


def _strip_base64_from_string(text: str) -> str:
    """Replace base64 image data URLs with placeholder.

    Matches the standard ToolOutputImage format:
    data:image/png;base64,{base64_data}
    """
    import re

    return re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
        "[base64 image removed]",
        text,
    )


def _strip_images_from_item(item: TResponseInputItem) -> TResponseInputItem:
    """Return copy of item with images replaced by placeholders.

    Preserves text content and replaces each image with "[image removed]".
    Handles both content field (user/assistant messages) and output field
    (function_call_output items which may contain base64 images).
    """
    if not isinstance(item, dict):
        return item

    result = dict(item)

    # Handle content field (user/assistant messages).
    content = item.get("content")
    if isinstance(content, list):
        new_content: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and _is_image_content(part):
                new_content.append({"type": "input_text", "text": "[image removed]"})
            else:
                new_content.append(part)
        result["content"] = new_content

    # Handle output field (function_call_output items).
    # Output can be a string (JSON) or a list (e.g., ToolOutputImage list).
    if item.get("type") == "function_call_output":
        output = item.get("output")
        if isinstance(output, str) and _contains_base64_image(output):
            result["output"] = _strip_base64_from_string(output)
        elif isinstance(output, list):
            # Handle list of items (e.g., from observe_scene returning ToolOutputImage).
            new_output: list[Any] = []
            for part in output:
                if isinstance(part, dict) and _is_image_content(part):
                    new_output.append({"type": "input_text", "text": "[image removed]"})
                else:
                    new_output.append(part)
            result["output"] = new_output

    return result


def _extract_text_from_turn(turn: Turn) -> str:
    """Extract all text content from a turn for summarization."""
    text_parts: list[str] = []

    for item in turn.items:
        if not isinstance(item, dict):
            continue

        role = item.get("role", "unknown")
        content = item.get("content")

        if isinstance(content, str):
            text_parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            item_texts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "input_text":
                        item_texts.append(part.get("text", ""))
                    elif part.get("type") == "text":
                        item_texts.append(part.get("text", ""))
                    elif _is_image_content(part):
                        item_texts.append("[image]")
            if item_texts:
                text_parts.append(f"{role}: {' '.join(item_texts)}")

        # Handle tool outputs (strip base64 images before extracting text).
        if item.get("type") == "function_call_output":
            output = item.get("output", "")
            call_id = item.get("call_id", "unknown")
            # Handle string outputs with embedded base64.
            if isinstance(output, str) and _contains_base64_image(output):
                output = _strip_base64_from_string(output)
                text_parts.append(f"tool_output({call_id}): {output}")
            # Handle list outputs (e.g., ToolOutputImage list from observe_scene).
            elif isinstance(output, list):
                n_images = sum(
                    1 for p in output if isinstance(p, dict) and _is_image_content(p)
                )
                if n_images > 0:
                    text_parts.append(f"tool_output({call_id}): [{n_images} images]")
                else:
                    text_parts.append(f"tool_output({call_id}): {output}")
            else:
                text_parts.append(f"tool_output({call_id}): {output}")

    return "\n".join(text_parts)


def _count_images_in_turn(turn: Turn) -> int:
    """Count total images in a turn (content images + tool output images)."""
    count = 0
    for item in turn.items:
        if not isinstance(item, dict):
            continue

        # Count images in content (user/assistant messages).
        content = item.get("content")
        if isinstance(content, list):
            count += sum(
                1
                for part in content
                if isinstance(part, dict) and _is_image_content(part)
            )

        # Count images in tool outputs.
        if item.get("type") == "function_call_output":
            output = item.get("output")
            if isinstance(output, str) and _contains_base64_image(output):
                count += 1
            elif isinstance(output, list):
                count += sum(
                    1 for p in output if isinstance(p, dict) and _is_image_content(p)
                )

    return count


def _compute_turn_hash(turn: Turn) -> str:
    """Compute a stable hash for a turn to use as cache key."""
    # Use the first item's content as the hash basis (stable within session).
    if not turn.items:
        return f"empty_{turn.start_index}"

    first_item = turn.items[0]
    content_str = json.dumps(first_item, sort_keys=True, default=str)
    return hashlib.md5(content_str.encode()).hexdigest()[:16]


class SummaryCache:
    """SQLite-based cache for turn summaries."""

    def __init__(self, db_path: Path):
        """Initialize the summary cache.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turn_summaries (
                turn_hash TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()

    def get(self, turn_hash: str) -> str | None:
        """Get a cached summary by turn hash."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                "SELECT summary FROM turn_summaries WHERE turn_hash = ?",
                (turn_hash,),
            )
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None

    def set(self, turn_hash: str, summary: str) -> None:
        """Cache a summary for a turn hash."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """
                INSERT OR REPLACE INTO turn_summaries (turn_hash, summary)
                VALUES (?, ?)
            """,
                (turn_hash, summary),
            )
            conn.commit()
            conn.close()

    def clear(self) -> None:
        """Clear all cached summaries."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("DELETE FROM turn_summaries")
            conn.commit()
            conn.close()


class TurnTrimmingSession:
    """Session wrapper that trims old turns by removing images and summarizing.

    This session wraps a SQLiteSession and provides memory-efficient context
    management by:
    1. Keeping the last N turns fully intact (with images)
    2. For older turns: removing ALL images and optionally summarizing via LLM

    The session implements the agents.Session protocol and can be used as a
    drop-in replacement for SQLiteSession.
    """

    def __init__(self, wrapped_session: SQLiteSession, cfg: DictConfig):
        """Initialize the turn-trimming session.

        Args:
            wrapped_session: The underlying SQLiteSession to wrap.
            cfg: Configuration with session_memory settings.
        """
        self.wrapped_session = wrapped_session
        self.session_id = wrapped_session.session_id
        self._cfg = cfg

        # Extract config values.
        memory_cfg = cfg.session_memory
        self._keep_last_n_turns: int = memory_cfg.keep_last_n_turns
        self._enable_summarization: bool = memory_cfg.enable_summarization
        self._summarization_model: str = memory_cfg.summarization_model
        self._summarization_thinking: str = memory_cfg.summarization_thinking

        # Resolve "agent" model to actual model name.
        if self._summarization_model == "agent":
            self._summarization_model = cfg.openai.model

        # Initialize summary cache next to the wrapped session's database.
        db_path = wrapped_session.db_path
        if isinstance(db_path, str) and db_path != ":memory:":
            cache_path = Path(db_path).parent / f"{self.session_id}_summaries.db"
        elif isinstance(db_path, Path):
            cache_path = db_path.parent / f"{self.session_id}_summaries.db"
        else:
            cache_path = Path(tempfile.gettempdir()) / f"{self.session_id}_summaries.db"

        self._summary_cache = SummaryCache(cache_path)
        self._openai_client: AsyncOpenAI | None = None

    def _get_openai_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client for summarization."""
        if self._openai_client is None:
            timeout = None
            if hasattr(self._cfg, "api_timeout"):
                timeout_cfg = self._cfg.api_timeout
                timeout = Timeout(
                    connect=timeout_cfg.connect,
                    read=timeout_cfg.read,
                    write=timeout_cfg.write,
                    pool=timeout_cfg.pool,
                )
            self._openai_client = (
                AsyncOpenAI(timeout=timeout) if timeout is not None else AsyncOpenAI()
            )
        return self._openai_client

    async def _summarize_turn(self, turn: Turn, turn_number: int) -> str:
        """Summarize turn text content via LLM.

        Args:
            turn: The turn to summarize.
            turn_number: The sequential turn number (0-indexed).

        Returns:
            Summary text for the turn.
        """
        # Check cache first.
        turn_hash = _compute_turn_hash(turn)
        cached = self._summary_cache.get(turn_hash)
        if cached is not None:
            console_logger.debug(f"Using cached summary for turn {turn_number}")
            return cached

        # Extract text and summarize.
        text = _extract_text_from_turn(turn)
        if not text.strip():
            summary = "[Empty turn]"
            self._summary_cache.set(turn_hash, summary)
            return summary

        console_logger.info(
            f"Summarizing turn {turn_number} with {self._summarization_model}"
        )

        # Build reasoning parameter if thinking is enabled.
        reasoning = None
        if self._summarization_thinking and self._summarization_thinking != "none":
            reasoning = Reasoning(effort=self._summarization_thinking)

        # Load summarization prompt from prompt registry.
        summarization_prompt = prompt_registry.get_prompt(
            prompt_enum=SessionMemoryPrompts.TURN_SUMMARIZATION
        )

        try:
            response = await self._get_openai_client().responses.create(
                model=self._summarization_model,
                instructions=summarization_prompt,
                input=text,
                reasoning=reasoning,
            )
            summary = response.output_text or "[Summary generation failed]"
        except Exception as e:
            console_logger.error(f"Summarization failed: {e}")
            # Fallback: just strip images without summarizing.
            summary = (
                f"[Turn {turn_number}: summarization failed, original text "
                f"truncated]\n{text[:500]}"
            )

        # Log context reduction achieved.
        n_images = _count_images_in_turn(turn)
        original_len = len(text)
        summary_len = len(summary)
        text_reduction = (
            ((original_len - summary_len) / original_len * 100)
            if original_len > 0
            else 0.0
        )
        console_logger.info(
            f"Turn {turn_number} summarized: "
            f"text {original_len:,} → {summary_len:,} chars "
            f"({text_reduction:.0f}% reduction), "
            f"{n_images} images removed"
        )

        self._summary_cache.set(turn_hash, summary)
        return summary

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Retrieve conversation history with old turns trimmed.

        Returns items with:
        - System messages always preserved intact
        - Pre-turn preamble preserved intact
        - Last N turns preserved intact (with images)
        - Older turns: images stripped, optionally summarized

        Args:
            limit: Maximum number of items to retrieve (applied after trimming).

        Returns:
            List of trimmed/processed items.
        """
        # Get all items from wrapped session.
        all_items = await self.wrapped_session.get_items()

        if not all_items:
            return []

        # Separate system messages (always keep intact).
        system_items: list[TResponseInputItem] = []
        non_system_items: list[TResponseInputItem] = []

        for item in all_items:
            if _is_system_message(item):
                system_items.append(item)
            else:
                non_system_items.append(item)

        # Parse turns from non-system items.
        turns, first_turn_start = _parse_turns(non_system_items)

        # Preserve preamble (items before first user message).
        preamble = non_system_items[:first_turn_start]

        # Split turns into old and recent.
        if len(turns) <= self._keep_last_n_turns:
            # Not enough turns to trim.
            console_logger.info(
                f"Session {self.session_id}: {len(turns)} turns, "
                f"keeping last {self._keep_last_n_turns}, "
                f"trimmed 0 turns, removed ~0 images"
            )
            result = system_items + non_system_items
            if limit is not None:
                result = result[-limit:]
            return result

        old_turns = turns[: -self._keep_last_n_turns]
        recent_turns = turns[-self._keep_last_n_turns :]

        # Process old turns.
        processed_old: list[TResponseInputItem] = []
        for turn_number, turn in enumerate(old_turns):
            if self._enable_summarization:
                # Summarize the turn into a single assistant message.
                # Using assistant role since the summary describes what happened
                # (including the assistant's actions and responses).
                summary = await self._summarize_turn(turn=turn, turn_number=turn_number)
                processed_old.append(
                    {
                        "role": "assistant",
                        "content": f"[Previous turn summary]\n{summary}",
                    }
                )
            else:
                # Just strip images from each item.
                for item in turn.items:
                    processed_old.append(_strip_images_from_item(item))

        # Collect recent turns (preserved intact).
        recent_items: list[TResponseInputItem] = []
        for turn in recent_turns:
            recent_items.extend(turn.items)

        # Combine: system + preamble + processed_old + recent.
        result = system_items + preamble + processed_old + recent_items

        # Log trimming stats.
        n_old_turns = len(old_turns)
        n_content_images = sum(
            1
            for turn in old_turns
            for item in turn.items
            if isinstance(item, dict) and isinstance(item.get("content"), list)
            for part in item["content"]
            if isinstance(part, dict) and _is_image_content(part)
        )
        # Count images in tool outputs (both string and list formats).
        n_tool_output_images = 0
        for turn in old_turns:
            for item in turn.items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "function_call_output":
                    continue
                output = item.get("output")
                if isinstance(output, str) and _contains_base64_image(output):
                    n_tool_output_images += 1
                elif isinstance(output, list):
                    n_tool_output_images += sum(
                        1
                        for p in output
                        if isinstance(p, dict) and _is_image_content(p)
                    )
        n_images_removed = n_content_images + n_tool_output_images
        console_logger.info(
            f"Session {self.session_id}: {len(turns)} turns, "
            f"keeping last {self._keep_last_n_turns}, "
            f"trimmed {n_old_turns} turns, removed ~{n_images_removed} images"
        )

        if limit is not None:
            result = result[-limit:]

        return result

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Add new items to the conversation history.

        Items are stored in the wrapped session unchanged. Trimming is applied
        lazily when get_items() is called.

        Args:
            items: List of input items to add.
        """
        await self.wrapped_session.add_items(items)

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent item from the session.

        Returns:
            The most recent item if it exists, None if empty.
        """
        return await self.wrapped_session.pop_item()

    async def clear_session(self) -> None:
        """Clear all items and cached summaries."""
        await self.wrapped_session.clear_session()
        self._summary_cache.clear()
