"""
Agent-based autoposting service.

The agent creates a plan, executes tools step by step,
and generates the final post text.

All in one continuous conversation (user-assistant-user-assistant...).
"""

import json
import logging
import time
import random
import re
from typing import Any

from services.database import Database
from services.llm import LLMClient
from services.twitter import TwitterClient
from tools.registry import TOOLS, get_tools_description
from config.personality import SYSTEM_PROMPT
from config.prompts.agent_autopost import AUTOPOST_AGENT_PROMPT
from config.schemas import PLAN_SCHEMA, POST_TEXT_SCHEMA, TOOL_REACTION_SCHEMA

logger = logging.getLogger(__name__)

# ⚠️ FALLBACK_TWEETS LEFT UNCHANGED (per your request)
from config.fallbacks import FALLBACK_TWEETS  # or keep your existing import


# -----------------------------
# HARD SANITIZER (FINAL GUARD)
# -----------------------------
def sanitize_post_text(text: str) -> str:
    if not text:
        return ""

    cleaned_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("[image:"):
            continue
        if "[" in s and "]" in s:
            continue
        cleaned_lines.append(s)

    text = " ".join(cleaned_lines)

    # Strip emojis / symbols
    text = re.sub(r"[^\w\s.,—']", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_agent_system_prompt() -> str:
    tools_desc = get_tools_description()
    return AUTOPOST_AGENT_PROMPT.format(tools_desc=tools_desc)


class AutoPostService:
    """Agent-based autoposting service with defensive execution."""

    def __init__(self, db: Database, tier_manager=None):
        self.db = db
        self.llm = LLMClient()
        self.twitter = TwitterClient()
        self.tier_manager = tier_manager

    def _sanitize_plan(self, plan: list[dict]) -> list[dict]:
        """
        Allow only known tools.
        Image tools may exist, but output is always sanitized later.
        """
        if not isinstance(plan, list):
            logger.warning("[AUTOPOST] Plan invalid — stripping")
            return []

        sanitized = []
        for step in plan:
            if not isinstance(step, dict):
                continue
            tool = step.get("tool")
            params = step.get("params", {})
            if tool not in TOOLS:
                logger.warning(f"[AUTOPOST] Unknown tool: {tool}")
                continue
            sanitized.append({"tool": tool, "params": params})
            if len(sanitized) >= 3:
                break

        return sanitized

    async def run(self) -> dict[str, Any]:
        start_time = time.time()
        logger.info("[AUTOPOST] === Starting ===")

        try:
            # -----------------------------
            # Tier gate
            # -----------------------------
            if self.tier_manager:
                can_post, reason = self.tier_manager.can_post()
                if not can_post:
                    logger.warning(f"[AUTOPOST] Blocked: {reason}")
                    return {"success": False, "error": reason}

            # -----------------------------
            # Load context
            # -----------------------------
            previous_posts = await self.db.get_recent_posts_formatted(limit=50)

            system_prompt = SYSTEM_PROMPT + get_agent_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Create a Twitter post.\n"
                        "Do not repeat previous posts.\n\n"
                        f"{previous_posts}\n\n"
                        "Create a plan if needed."
                    ),
                },
            ]

            # -----------------------------
            # Planning
            # -----------------------------
            plan_raw = await self.llm.chat(messages, PLAN_SCHEMA)
            try:
                plan_data = json.loads(plan_raw) if isinstance(plan_raw, str) else plan_raw
            except Exception:
                plan_data = {}

            plan = self._sanitize_plan(plan_data.get("plan", []))
            messages.append({"role": "assistant", "content": json.dumps(plan_data)})

            # -----------------------------
            # Tool execution
            # -----------------------------
            image_bytes = None
            for step in plan:
                tool = step["tool"]
                params = step["params"]

                if tool == "generate_image":
                    try:
                        image_bytes = await TOOLS[tool](params.get("prompt", ""))
                        messages.append({"role": "user", "content": "Tool result: image generated"})
                    except Exception as e:
                        logger.error(f"[AUTOPOST] Image tool failed: {e}")
                        image_bytes = None
                else:
                    try:
                        result = await TOOLS[tool](**params)
                        messages.append({"role": "user", "content": f"Tool result: {result}"})
                    except Exception as e:
                        logger.error(f"[AUTOPOST] Tool {tool} failed: {e}")

                reaction = await self.llm.chat(messages, TOOL_REACTION_SCHEMA)
                messages.append({"role": "assistant", "content": reaction.get("thinking", "")})

            # -----------------------------
            # Final tweet generation
            # -----------------------------
            messages.append({
                "role": "user",
                "content": (
                    "Write ONE tweet.\n"
                    "Text only.\n"
                    "No image descriptions.\n"
                    "No brackets or metadata.\n"
                    "Quiet observational tone.\n"
                    "Output the tweet text only."
                ),
            })

            post_raw = await self.llm.chat(messages, POST_TEXT_SCHEMA)

            if isinstance(post_raw, dict):
                post_text = post_raw.get("post_text", "")
            else:
                try:
                    post_text = json.loads(post_raw).get("post_text", "")
                except Exception:
                    post_text = post_raw or ""

            # -----------------------------
            # SANITIZE + GUARANTEE
            # -----------------------------
            post_text = sanitize_post_text(post_text)

            if not post_text or len(post_text) < 20:
                logger.warning("[AUTOPOST] Invalid text — forcing fallback")
                post_text = random.choice(FALLBACK_TWEETS)

            if "[image" in post_text.lower():
                logger.error("[AUTOPOST] Image leak detected — forcing fallback")
                post_text = random.choice(FALLBACK_TWEETS)

            # -----------------------------
            # Upload image (optional)
            # -----------------------------
            media_ids = None
            if image_bytes:
                try:
                    media_id = await self.twitter.upload_media(image_bytes)
                    media_ids = [media_id]
                except Exception as e:
                    logger.error(f"[AUTOPOST] Media upload failed: {e}")
                    media_ids = None

            # -----------------------------
            # POST WITH RETRY (NO MISS)
            # -----------------------------
            tweet_data = None
            for attempt in range(3):
                try:
                    tweet_data = await self.twitter.post(post_text, media_ids=media_ids)
                    break
                except Exception as e:
                    logger.error(f"[AUTOPOST] Twitter post failed ({attempt+1}/3): {e}")
                    time.sleep(5)

            if not tweet_data:
                logger.critical("[AUTOPOST] Final retry failed — posting fallback")
                post_text = random.choice(FALLBACK_TWEETS)
                tweet_data = await self.twitter.post(post_text)

            # -----------------------------
            # Save + return
            # -----------------------------
            await self.db.save_post(post_text, tweet_data["id"], image_bytes is not None)

            duration = round(time.time() - start_time, 1)
            logger.info(f"[AUTOPOST] === Completed in {duration}s ===")

            return {
                "success": True,
                "tweet_id": tweet_data["id"],
                "text": post_text,
                "duration": duration,
            }

        except Exception as e:
            logger.exception("[AUTOPOST] Fatal error")
            return {"success": False, "error": str(e)}
