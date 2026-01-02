"""
Twitter API Tier Manager.

Tier detection is DISABLED.
The system runs in fail-open mode using static Free-tier defaults.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


# Tier definitions based on Twitter API pricing
TIER_FEATURES = {
    "free": {
        "mentions": False,
        "post_limit": 500,
        "read_limit": 100,
        "daily_post_limit": 15,
        "daily_reply_limit": 0
    }
}


class TierManager:
    """
    Tier detection is disabled.

    Behavior:
    - Always initialized
    - Always uses Free-tier limits
    - Never calls Twitter Usage API
    - Never pauses automatically
    """

    def __init__(self, db=None):
        self.db = db

        # Static tier state
        self.tier: str = "free"
        self.project_cap: int = 0
        self.project_usage: int = 0
        self.cap_reset_day: int | None = None

        self.last_tier_check: datetime | None = None
        self.tier_check_interval = timedelta(hours=1)

        self.is_initialized = True
        self.is_paused = False
        self.pause_reason: str | None = None

    async def initialize(self) -> dict[str, Any]:
        """
        Initialize tier manager (static).
        """
        logger.info("[TIER] Tier detection disabled — using static FREE tier")
        self._log_status()

        return {
            "tier": self.tier,
            "method": "static_disabled"
        }

    async def detect_tier(self) -> dict[str, Any]:
        """
        Disabled tier detection (no-op).
        """
        return {
            "tier": self.tier,
            "method": "disabled"
        }

    async def refresh_usage(self) -> None:
        """
        Disabled.
        """
        return

    async def maybe_refresh_tier(self) -> None:
        """
        Disabled.
        """
        return

    def get_usage_percent(self) -> float:
        return 0.0

    def can_post(self) -> tuple[bool, str | None]:
        """
        Posting is always allowed unless manually paused.
        """
        if self.is_paused:
            return False, self.pause_reason
        return True, None

    def can_use_mentions(self) -> tuple[bool, str | None]:
        """
        Mentions depend ONLY on settings.
        """
        if not settings.allow_mentions:
            return False, "mentions_disabled_in_settings"

        features = TIER_FEATURES["free"]
        if not features.get("mentions", False):
            return False, "mentions_not_available_on_free_tier"

        return True, None

    def get_daily_limits(self) -> tuple[int, int]:
        features = TIER_FEATURES["free"]
        return (
            features.get("daily_post_limit", 15),
            features.get("daily_reply_limit", 0)
        )

    def resume(self) -> None:
        self.is_paused = False
        self.pause_reason = None
        logger.info("[TIER] Operations resumed")

    def _log_status(self) -> None:
        features = TIER_FEATURES["free"]

        logger.info("=" * 50)
        logger.info("[TIER] Tier detection DISABLED")
        logger.info("[TIER] Using FREE tier defaults")
        logger.info(f"[TIER] Mentions available: {features.get('mentions', False)}")
        logger.info("=" * 50)

    def get_status(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "is_initialized": True,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "project_cap": 0,
            "project_usage": 0,
            "usage_percent": 0.0,
            "cap_reset_day": None,
            "features": TIER_FEATURES["free"],
            "last_check": None
        }
