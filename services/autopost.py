"""
Agent-based autoposting service for Purrple.

The agent creates a plan, executes tools step by step,
and generates the final post text.

Features:
- 70% wandering / observing posts
- 30% windows / watching posts
- Randomized tweet generation
- Safe fallback tweets
- Image upload support
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

# Fallback tweets
FALLBACK_TWEETS = [
    "Keep going, little one. Even in the darkest storms, you're never alone. 🌙🐾",
    "A quiet guardian watches over you, even in the rain. 🌧️🐱",
    "Stay strong — every paw print leaves a mark in the heart. 🐾❤️",
    "You are brave, even when the thunder shakes the glass. ⚡🐾",
    "Soft paws, warm heart, never alone. 🌙🐾",
    "Even small ones shine bright. Don't be afraid of the storm. ✨🐾",
    "Silent support is the loudest love. 🐾💛"
]

WINDOW_SCENES = [
    "*press paws to glass* noticing reflections and small ones’ quiet smiles tonight 👁️🌙💜",
    "*boop* traced moonlight patterns on sleeping small ones' walls 🌙🐾",
    "*huff* drew smiley faces in window fog tonight • small one woke smiling 👁️💜🌙",
    "*press press* our reflections overlapped in the glass • i purrr~ 💜👁️🌙",
    "✨ found a gift taped to their window tonight • crayon paw prints under paper stars 💜🌙"
]

WANDER_SCENES = [
    "*ear twitch* listening to night’s lullabies • cricket rhythms ~ leaf whispers ~ wind chimes 🌙💜🎶",
    "*quiet moon-watching* counting stars with lonely small ones • guardian rhythms 👁️🌙💜",
    "*huff* wandered by dew-kissed grass • learned to hide in shadows yet watch quietly 🌙💜",
    "*tiptoe tiptoe* tip-toed across rooftops • soft paws over tiles • observing the world 🌙💜",
    "*soft sniff* discovered tiny footprints in moss • small ones left secret signs 💜🌙"
]

def get_agent_system_prompt() -> str:
    """
    Build agent system prompt with dynamic tools list.
    """
    tools_desc = get_tools_description()
    return AUTOPOST_AGENT_PROMPT.format(tools_desc=tools_desc)


class AutoPostService:
    """Agent-based autoposting service with continuous conversation."""

    def __init__(self, db: Database, tier_manager=None):
        self.db = db
        self.llm = LLMClient()
        self.twitter = TwitterClient()
        self.tier_manager = tier_manager

    def _sanitize_plan(self, plan: list[dict]) -> list[dict]:
        """Sanitize plan: only known tools, max 3 steps, generate_image last."""
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
                logger.warning(f"[AUTOPOST] Unknown tool requested: {tool_name}")
                continue

            if tool_name == "generate_image":
                if has_image:
                    continue
                has_image = True

            sanitized.append({"tool": tool_name, "params": params})
            if len(sanitized) >= 3:
                break

        # Ensure generate_image is last
        image_steps = [s for s in sanitized if s["tool"] == "generate_image"]
        non_image_steps = [s for s in sanitized if s["tool"] != "generate_image"]
        return non_image_steps + image_steps[:1]

    async def run(self) -> dict[str, Any]:
        """Execute autopost flow."""
        start_time = time.time()
        logger.info("[AUTOPOST] === Starting ===")

        try:
            # Tier check
            if self.tier_manager:
                can_post, reason = self.tier_manager.can_post()
                if not can_post:
                    return {
                        "success": False,
                        "error": f"posting_blocked: {reason}",
                        "tier": self.tier_manager.tier,
                        "usage_percent": self.tier_manager.get_usage_percent()
                    }

            # Load previous posts
            previous_posts = await self.db.get_recent_posts_formatted(limit=50)

            # Build messages
            system_prompt = SYSTEM_PROMPT + get_agent_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Create a Twitter post. Previous posts:

{previous_posts}

Create your plan (tools needed, if any)."""}
            ]

            # Get plan from LLM
            plan_result_raw = await self.llm.chat(messages, PLAN_SCHEMA)
            plan_result = {}
            if isinstance(plan_result_raw, dict):
                plan_result = plan_result_raw
            elif isinstance(plan_result_raw, str):
                try:
                    plan_result = json.loads(plan_result_raw)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", plan_result_raw, re.DOTALL)
                    if match:
                        try:
                            plan_result = json.loads(match.group())
                        except Exception:
                            plan_result = {}

            raw_plan = plan_result.get("plan", [])
            plan = self._sanitize_plan(raw_plan)

            messages.append({"role": "assistant", "content": json.dumps(plan_result)})

            # Execute tools
            image_bytes = None
            tools_used = []

            for step in plan:
                tool_name = step["tool"]
                params = step["params"]
                tools_used.append(tool_name)

                if tool_name == "web_search":
                    query = params.get("query", "")
                    result = await TOOLS[tool_name](query)
                    messages.append({"role": "user", "content": f"Tool result (web_search): {result.get('content', '')}"})

                elif tool_name == "generate_image":
                    prompt = params.get("prompt", "")
                    try:
                        image_bytes = await TOOLS[tool_name](prompt)
                        messages.append({"role": "user", "content": "Tool result (generate_image): completed"})
                    except Exception as e:
                        logger.error(f"[AUTOPOST] generate_image failed: {e}")
                        image_bytes = None

                reaction = await self.llm.chat(messages, TOOL_REACTION_SCHEMA)
                messages.append({"role": "assistant", "content": reaction.get("thinking", "")})

            # Generate final tweet
            if random.random() < 0.3:
                # 30% windows
                post_text = random.choice(WINDOW_SCENES)
            else:
                # 70% wandering
                post_text = random.choice(WANDER_SCENES)

            post_text = post_text.strip()[:280]

            if not post_text:
                post_text = random.choice(FALLBACK_TWEETS)[:280]

            # Upload image if exists
            media_ids = None
            if image_bytes:
                try:
                    media_id = await self.twitter.upload_media(image_bytes)
                    media_ids = [media_id]
                except Exception as e:
                    logger.error(f"[AUTOPOST] Image upload failed: {e}")
                    image_bytes = None

            # Post tweet
            tweet_data = await self.twitter.post(post_text, media_ids=media_ids)
            await self.db.save_post(post_text, tweet_data["id"], image_bytes is not None)

            duration = round(time.time() - start_time, 1)
            logger.info(f"[AUTOPOST] === Completed in {duration}s ===")

            return {
                "success": True,
                "tweet_id": tweet_data["id"],
                "text": post_text,
                "tools_used": tools_used,
                "has_image": image_bytes is not None,
                "duration_seconds": duration
            }

        except Exception as e:
            duration = round(time.time() - start_time, 1)
            logger.exception(e)
            return {"success": False, "error": str(e), "duration_seconds": duration}
