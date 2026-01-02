"""
Twitter API Tier Manager.

Handles automatic tier detection, usage tracking, and limit management.
Provides checks before API calls to prevent hitting limits.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


TIER_CAPS = {
    "free": 100,
    "basic": 10_000,
    "pro": 1_000_000,
    "enterprise": 10_000_000
}

TIER_FEATURES = {
    "free": {
        "mentions": False,
        "post_limit": 500,
        "read_limit": 100,
        "daily_post_limit": 15,
        "daily_reply_limit": 0
    },
    "basic": {
        "mentions": True,
        "post_limit": 3_000,
        "read_limit": 10_000,
        "daily_post_limit": 50,
        "daily_reply_limit": 50
    },
    "pro": {
        "mentions": True,
        "post_limit": 300_000,
        "read_limit": 1_000_000,
        "daily_post_limit": 500,
        "daily_reply_limit": 500
    },
    "enterprise": {
        "mentions": True,
        "post_limit": None,
        "read_limit": 10_000_000,
        "daily_post_limit": 1000,
        "daily_reply_limit": 1000
    }
}


class TierManager:
    """
    Manages Twitter API tier detection and usage tracking.
    """

    def __init__(self, db=None):
        self.db = db

        # Default SAFE tier
        self.tier: str = "free"
        self.project_id: str | None = None

        self.project_cap: int = 0
        self.project_usage: int = 0
        self.cap_reset_day: int | None = None

        self.last_tier_check: datetime | None = None

        # Cooldown to avoid rate limits
        self.tier_check_interval = timedelta(hours=6)

        self.is_initialized = False
        self.is_paused = False
        self.pause_reason: str | None = None

    async def initialize(self) -> dict[str, Any]:
        logger.info("[TIER] Initializing tier manager...")
        result = await self.detect_tier()
