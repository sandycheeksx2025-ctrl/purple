"""
Twitter API Tier Manager.

Handles automatic tier detection, usage tracking, and limit management.
Provides checks before API calls to prevent hitting limits.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


# Tier definitions based on Twitter API pricing
TIER_CAPS = {
    "free": 100,          # Tweet Read Cap
    "basic": 10_000,
    "pro": 1_000_000,
    "enterprise": 10_000_000
}

TIER_FEATURES = {
    "free": {
        "mentions": False,
        "post_limit": 500,           # Posts per month
        "read_limit": 100,
        "daily_post_limit": 15,      # ~500/30
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

        self.tier: str | None = None
        self.project_id: str | None = None

        self.project_cap: int = 0
        self.project_usage: int = 0
        self.cap_reset_day: int | None = None

        self.rate_limit_limit: int = 0
        self.rate_limit_remaining: int = 0
        self.rate_limit_reset: datetime | None = None

        self.last_tier_check: datetime | None = None
        self.tier_check_interval = timedelta(hours=1)

        self.is_initialized = False
        self.is_paused = False
        self.pause_reason: str | None = None

    async def initialize(self) -> dict[str, Any]:
        logger.info("[TIER] Initializing tier manager...")

        result = await self.detect_tier()

        if result.get("tier") != "unknown":
            self.is_initialized = True
            self._log_status()
        else:
            logger.warning("[TIER] Could not detect tier, some features may not work correctly")

        return result

    async def detect_tier(self) -> dict[str, Any]:
        try:
            url = "https://api.twitter.com/2/usage/tweets"
            headers = {
                "Authorization": f"Bearer {settings.twitter_bearer_token}"
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)

                # ✅ SAFE 429 HANDLING
                if response.status_code == 429:
                    logger.warning("[TIER] Rate limited by Usage API (429). Keeping existing tier.")
                    self.last_tier_check = datetime.now()
                    return {
                        "tier": self.tier or "unknown",
                        "rate_limited": True
                    }

                if response.status_code == 403:
                    self.tier = "free"
                    logger.info("[TIER] Usage API returned 403, assuming Free tier")
                    return {
                        "tier": "free",
                        "method": "usage_api_403"
                    }

                response.raise_for_status()
                data = response.json()

            usage_data = data.get("data", {})
            self.project_cap = int(usage_data.get("project_cap", 0))
            self.project_usage = int(usage_data.get("project_usage", 0))
            self.cap_reset_day = usage_data.get("cap_reset_day")
            self.project_id = usage_data.get("project_id")

            if self.project_cap >= 10_000_000:
                self.tier = "enterprise"
            elif self.project_cap >= 1_000_000:
                self.tier = "pro"
            elif self.project_cap >= 10_000:
                self.tier = "basic"
            elif self.project_cap <= 500:
                self.tier = "free"
            else:
                self.tier = "unknown"

            self.last_tier_check = datetime.now()
            self._check_usage_warnings()

            return {
                "tier": self.tier,
                "project_cap": self.project_cap,
                "project_usage": self.project_usage,
                "usage_percent": self.get_usage_percent(),
                "cap_reset_day": self.cap_reset_day,
                "features": TIER_FEATURES.get(self.tier, {})
            }

        except Exception as e:
            logger.error(f"[TIER] Error detecting tier: {e}")
            return {
                "tier": "unknown",
                "error": str(e)
            }

    async def refresh_usage(self) -> None:
        await self.detect_tier()

    async def maybe_refresh_tier(self) -> None:
        if self.last_tier_check is None:
            await self.detect_tier()
            return

        if datetime.now() - self.last_tier_check > self.tier_check_interval:
            logger.info("[TIER] Hourly tier check...")
            old_tier = self.tier
            await self.detect_tier()

            if old_tier != self.tier:
                logger.info(f"[TIER] Tier changed: {old_tier} -> {self.tier}")

    def get_usage_percent(self) -> float:
        if self.project_cap <= 0:
            return 0.0
        return (self.project_usage / self.project_cap) * 100

    def _check_usage_warnings(self) -> None:
        percent = self.get_usage_percent()

        if percent >= 100:
            self.is_paused = True
            self.pause_reason = "monthly_cap_reached"
            logger.error(
                f"[TIER] MONTHLY CAP REACHED "
                f"({self.project_usage}/{self.project_cap})."
            )
        elif percent >= 90:
            logger.warning(f"[TIER] Usage at {percent:.1f}%")
        elif percent >= 80:
            logger.warning(f"[TIER] Usage at {percent:.1f}%")

    def can_post(self) -> tuple[bool, str | None]:
        if not self.is_initialized:
            return True, None

        if self.is_paused:
            return False, self.pause_reason

        if self.get_usage_percent() >= 100:
            return False, "monthly_cap_reached"

        return True, None

    def can_use_mentions(self) -> tuple[bool, str | None]:
        if not settings.allow_mentions:
            return False, "mentions_disabled_in_settings"

        if not self.is_initialized:
            return True, None

        features = TIER_FEATURES.get(self.tier, {})
        if not features.get("mentions", False):
            return False, f"mentions_not_available_on_{self.tier}_tier"

        return True, None

    def get_daily_limits(self) -> tuple[int, int]:
        tier = self.tier or "free"
        features = TIER_FEATURES.get(tier, TIER_FEATURES["free"])
        return (
            features.get("daily_post_limit", 15),
            features.get("daily_reply_limit", 0)
        )

    def resume(self) -> None:
        self.is_paused = False
        self.pause_reason = None
        logger.info("[TIER] Operations resumed")

    def _log_status(self) -> None:
        features = TIER_FEATURES.get(self.tier, {})
        logger.info("=" * 50)
        logger.info(f"[TIER] Detected tier: {self.tier.upper()}")
        logger.info(
            f"[TIER] Read cap: "
            f"{self.project_usage}/{self.project_cap} "
            f"({self.get_usage_percent():.1f}%)"
        )
        logger.info(f"[TIER] Cap resets on day: {self.cap_reset_day}")
        logger.info(f"[TIER] Mentions available: {features.get('mentions', False)}")
        logger.info("=" * 50)

    def get_status(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "is_initialized": self.is_initialized,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "project_cap": self.project_cap,
            "project_usage": self.project_usage,
            "usage_percent": self.get_usage_percent(),
            "cap_reset_day": self.cap_reset_day,
            "rate_limit_remaining": self.rate_limit_remaining,
            "rate_limit_limit": self.rate_limit_limit,
            "features": TIER_FEATURES.get(self.tier, {}),
            "last_check": self.last_tier_check.isoformat() if self.last_tier_check else None
        }
