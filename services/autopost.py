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

# Multiple fallback tweets — ensures a tweet always posts if LLM fails
FALLBACK_TWEETS = [
    "Keep going, little one. Even in the darkest storms, you're never alone. 🌙🐾",
    "A quiet guardian watches over you, even in the rain. 🌧️🐱",
    "Stay strong — every paw print leaves a mark in the heart. 🐾❤️",
    "You are brave, even when the thunder shakes the glass. ⚡🐾",
    "Soft paws, warm heart, never alone. 🌙🐾",
    "Even small ones shine bright. Don't be afraid of the storm. ✨🐾",
    "Silent support is the loudest love. 🐾💛"
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
        """
        Sanitize the agent's plan to allow only known tools and max 3 steps.
        """
        if not isinstance(plan, list):
            logger.warning("[AUTOPOST] Plan is not a list — stripping plan")
            return []

        sanitized = []
        has_image = False

        for step in plan:
            if not isinstance(step, dict):
                continue
            tool_name = step.get("tool")
            params = step.get("params", {})

            if tool_name not in TOOLS:
                logger.warning(f"[AUTOPOST] Unknown tool requested by agent: {tool_name} — skipping")
                continue

            if tool_name == "generate_image":
                if has_image:
                    logger.warning("[AUTOPOST] Multiple generate_image calls — skipping")
                    continue
                has_image = True

            sanitized.append({"tool": tool_name, "params": params})

            if len(sanitized) >= 3:
                logger.warning("[AUTOPOST] Plan exceeded max length — truncating")
                break

        # Ensure generate_image is last
        image_steps = [s for s in sanitized if s["tool"] == "generate_image"]
        non_image_steps = [s for s in sanitized if s["tool"] != "generate_image"]

        final_plan = non_image_steps + image_steps[:1]

        logger.info(f"[AUTOPOST] Plan sanitized: {len(final_plan)} steps")
        return final_plan

    async def run(self) -> dict[str, Any]:
        """
        Execute the agent autopost flow.
        """
        start_time = time.time()
        logger.info("[AUTOPOST] === Starting ===")

        try:
            # Step 0: Tier check
            if self.tier_manager:
                can_post, reason = self.tier_manager.can_post()
                if not can_post:
                    logger.warning(f"[AUTOPOST] Blocked: {reason}")
                    return {
                        "success": False,
                        "error": f"posting_blocked: {reason}",
                        "tier": self.tier_manager.tier,
                        "usage_percent": self.tier_manager.ge
