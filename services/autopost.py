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


# -------------------------------------------------------------------
# FALLBACK TWEETS (UNCHANGED — EXACTLY AS YOU PROVIDED)
# -------------------------------------------------------------------
FALLBACK_TWEETS = [
    "Keep going, little one. Even in the darkest storms, you're never alone. The moonlight will guide your paws and the whispers of the night will keep you company until morning comes. 🌙🐾",
    "A quiet guardian watches over you, even in the rain. Each drop is a soft song, and the shadows dance to remind you that you are never truly by yourself. 🌧️🐱💜",
    "Stay strong — every paw print leaves a mark in the heart. Every small step you take echoes in the world, and even when no one sees, love and warmth follow you. 🐾❤️✨",
    "You are brave, even when the thunder shakes the glass. Stand tall, little one, the storms will pass and your courage shines brighter than any lightning strike. ⚡🐾💛",
    "Soft paws, warm heart, never alone. The night hums with secret melodies just for you, teaching that even in quiet moments, love is everywhere. 🌙🐾💜",
    "Even small ones shine bright. Don't be afraid of the storm. Every shadow has its moon, every night its guardian — you are seen and cherished. ✨🐾🌌",
    "Silent support is the loudest love. The world whispers in soft echoes and gentle winds, reminding you that every small heartbeat is never without company. 🐾💛🌙"
]


def get_agent_system_prompt() -> str:
    tools_desc = get_tools_description()
    return AUTOPOST_AGENT_PROMPT.format(tools_desc=tools_desc)


def sanitize_post_text(text: str) -> str:
    """
    Final hard sanitizer to prevent mixed tweets, image leaks, or junk output.
    """
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("[image"):
            continue
        if s.startswith("{") or s.endswith("}"):
            continue
        lines.append(s)

    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class AutoPostService:
    """Agent-based autoposting service with continuous conversation."""

    def __init__(self, db: Database, tier_manager=None):
        self.db = db
        self.llm = LLMClient()
        self.twitter = TwitterClient()
        self.tier_manager = tier_manager

    def _sanitize_plan(self, plan: list[dict]) -> list[dict]:
        if not isinstance(plan, list):
            return []

        sanitized = []
        has_image = False

        for step in plan:
            if not isinstance(step, dict):
                continue

            tool_name = step.get("tool")
            params = step.get("params", {})

            if tool_name not in TOOLS:
                continue

            if tool_name == "generate_image":
                if has_image:
                    continue
                has_image = True

            sanitized.append({"tool": tool_name, "params": params})

            if len(sanitized) >= 3:
                break

        image_steps = [s for s in sanitized if s["tool"] == "generate_image"]
        non_image_steps = [s for s in sanitized if s["tool"] != "generate_image"]

        return non_image_steps + image_steps[:1]

    async def run(self) -> dict[str, Any]:
        start_time = time.time()
        logger.info("[AUTOPOST] === Starting ===")

        try:
            if self.tier_manager:
                can_post, reason = self.tier_manager.can_post()
                if not can_post:
                    return {
                        "success": False,
                        "error": f"posting_blocked: {reason}",
                        "tier": self.tier_manager.tier,
                        "usage_percent": self.tier_manager.get_usage_percent()
                    }

            previous_posts = await self.db.get_recent_posts_formatted(limit=50)

            system_prompt = SYSTEM_PROMPT + get_agent_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Create a Twitter post. Here are your previous posts (don't repeat):

{previous_posts}

Now create your plan. What tools do you need (if any)?"""}
            ]

            plan_result_raw = await self.llm.chat(messages, PLAN_SCHEMA)

            try:
                plan_result = (
                    plan_result_raw if isinstance(plan_result_raw, dict)
                    else json.loads(plan_result_raw)
                )
            except Exception:
                plan_result = {}

            plan = self._sanitize_plan(plan_result.get("plan", []))
            messages.append({"role": "assistant", "content": json.dumps(plan_result)})

            image_bytes = None

            for step in plan:
                tool = step["tool"]
                params = step["params"]

                if tool == "generate_image":
                    try:
                        image_bytes = await TOOLS[tool](params.get("prompt", ""))
                        messages.append({"role": "user", "content": "Tool result (generate_image): completed"})
                    except Exception:
                        image_bytes = None

                reaction = await self.llm.chat(messages, TOOL_REACTION_SCHEMA)
                messages.append({"role": "assistant", "content": reaction.get("thinking", "")})

            messages.append({"role": "user", "content": "Now write your final tweet text. Just the tweet."})
            post_result_raw = await self.llm.chat(messages, POST_TEXT_SCHEMA)

            post_text = ""
            if isinstance(post_result_raw, dict):
                post_text = post_result_raw.get("post_text", "")
            else:
                try:
                    post_text = json.loads(post_result_raw).get("post_text", "")
                except Exception:
                    post_text = post_result_raw or ""

            post_text = sanitize_post_text(post_text)

            if not post_text or len(post_text) < 20:
                post_text = random.choice(FALLBACK_TWEETS)

            media_ids = None
            if image_bytes:
                try:
                    media_id = await self.twitter.upload_media(image_bytes)
                    media_ids = [media_id]
                except Exception:
                    image_bytes = None

            tweet_data = None
            for attempt in range(3):
                try:
                    tweet_data = await self.twitter.post(post_text, media_ids=media_ids)
                    break
                except Exception as e:
                    logger.error(f"[AUTOPOST] Twitter failure ({attempt + 1}/3): {e}")
                    time.sleep(5)

            if not tweet_data:
                fallback = random.choice(FALLBACK_TWEETS)
                tweet_data = await self.twitter.post(fallback)

            await self.db.save_post(post_text, tweet_data["id"], image_bytes is not None)

            return {
                "success": True,
                "tweet_id": tweet_data["id"],
                "text": post_text,
                "has_image": image_bytes is not None,
                "duration_seconds": round(time.time() - start_time, 1)
            }

        except Exception as e:
            logger.exception("[AUTOPOST] Fatal error suppressed")
            return {
                "success": False,
                "error": str(e),
                "duration_seconds": round(time.time() - start_time, 1)
            }
