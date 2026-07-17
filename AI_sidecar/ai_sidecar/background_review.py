"""Alias for post_action_review — the plan specified background_review.py name.

This module re-exports post_action_review for backwards compatibility.
"""
from ai_sidecar.autonomy.post_action_review import review_action, review_heal_strategy

__all__ = ["review_action", "review_heal_strategy"]
