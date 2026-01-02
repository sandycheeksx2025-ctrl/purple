"""
Twitter API Tier Manager.
Safe, non-fatal tier detection with cooldown.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import httpx
from config.settings import settings

logger = logging.getLogger(__name__)


class TierManager:
    def __init__(self, db=None):
        self.db = db

        # SAFE defaults
        self.tier: str = "free"
        self.project_cap: int = 0
        self.project_usage: int = 0
        self.cap_reset_day: int | None = None

        self.last_tier_check: datetime | None = None
        self.tier_check_interval = timedelta(hours=6)

        self.is_initialized = False
        self.is_paused = False
        self.pause_reason: str | None = None

    async def initialize(self) -> dict[str, Any]:
        logger.info("[TIER] Initializing tier manager...")
        result = await self.detect_tier()
        self.is_initialized = True
        return result

    async def detect_tier(self) -> dict[str, Any]:
        # Cooldown guard
        if self.last_tier_check and datetime.now() - self.last_tier_check < self.tier_check_interval:
            logger.info("[TIER] Skipping tier detection (cooldown active)")
            return {"tier": self.tier, "method": "cached"}

        self.last_tier_check = datetime.now()

        try:
            url = "https://api.twitter.com/2/usage/tweets"
            headers = {
                "Authorization": f"Bearer {settings.twitter_bearer_token}"
            }

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, headers=headers)

                if response.status_code == 429:
                    logger.warning("[TIER] Rate limited — keeping existing tier")
                    return {"tier": self.tier, "method": "rate_limited"}

                if response.status_code == 403:
                    logger.info("[TIER] Usage API 403 — assuming Free tier")
                    self.tier = "free"
                    return {"tier": "free", "method": "403"}

                response.raise_for_status()
                data = response.json()

            usage = data.get("data", {})
            self.project_cap = int(usage.get("project_cap", 0))
            self.project_usage = int(usage.get("project_usage", 0))
            self.cap_reset_day = usage.get("cap_reset_day")

            if self.project_cap >= 1_000_000:
                self.tier = "pro"
            elif self.project_cap >= 10_000:
                self.tier = "basic"
            else:
                self.tier = "free"

            return {
                "tier": self.tier,
                "project_cap": self.project_cap,
                "project_usage": self.project_usage,
                "usage_percent": self.get_usage_percent(),
            }

        except Exception as e:
            logger.warning(f"[TIER] Detection failed — continuing safely ({e})")
            return {"tier": self.tier, "method": "error"}

    async def maybe_refresh_tier(self) -> None:
        await self.detect_tier()

    def get_usage_percent(self) -> float:
        if self.project_cap <= 0:
            return 0.0
        return (self.project_usage / self.project_cap) * 100

    def can_post(self) -> tuple[bool, str | None]:
        if self.is_paused:
            return False, self.pause_reason
        return True, None
