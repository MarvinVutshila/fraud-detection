"""
services/decision_service.py
============================
Pure business‑logic layer that converts a fraud probability into:
  * decision   — APPROVE | REVIEW | BLOCK
  * risk_level — LOW | MEDIUM | HIGH | CRITICAL

The thresholds are pulled from config but can be overridden per instance
(useful for A/B testing or tenant‑specific rules).

Decision rules
──────────────
  prob < approve_threshold              → APPROVE
  approve_threshold <= prob < block_threshold → REVIEW
  prob >= block_threshold               → BLOCK

Risk level rules (independent of decision)
──────────────────────────────────────────
  prob < 0.20  → LOW
  prob < 0.50  → MEDIUM
  prob < 0.80  → HIGH
  prob >= 0.80 → CRITICAL

Model threshold (e.g., 0.9244) is NOT used here – it belongs only in:
  * model evaluation metrics (precision/recall)
  * internal dashboards (optional binary flag)
  * NOT in live decision making.
"""

from __future__ import annotations

import logging
from typing import Optional

from fraud_detection.core.config import APPROVE_THRESHOLD, BLOCK_THRESHOLD

logger = logging.getLogger(__name__)


class DecisionService:
    """
    Converts a raw probability into a structured decision and risk level.

    Parameters
    ----------
    approve_threshold : float
        Probabilities strictly below this are approved automatically.
    block_threshold   : float
        Probabilities at or above this are blocked automatically.
        Values in [approve_threshold, block_threshold) go to REVIEW.
    model_threshold   : float, optional
        Used ONLY for generating a binary `is_fraud` flag (e.g., for metrics or UI).
        If None, `is_fraud` will be derived from decision (BLOCK → True, else False).
    """

    def __init__(
        self,
        approve_threshold: float = APPROVE_THRESHOLD,
        block_threshold:   float = BLOCK_THRESHOLD,
        model_threshold:   Optional[float] = None,
    ) -> None:
        if not (0.0 <= approve_threshold < block_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: approve={approve_threshold}, block={block_threshold}. "
                "Must satisfy 0 <= approve < block <= 1."
            )
        self.approve_threshold = approve_threshold
        self.block_threshold   = block_threshold
        self.model_threshold   = model_threshold   # for binary flag only

        logger.info(
            "DecisionService ready | approve<%.2f | review=[%.2f,%.2f) | block>=%.2f",
            approve_threshold, approve_threshold, block_threshold, block_threshold,
        )
        if model_threshold is not None:
            logger.info("Model threshold (binary flag only) = %.4f", model_threshold)

    # ── Public business methods ─────────────────────────────────────────────

    def get_decision(self, probability: float) -> str:
        """
        Return the routing decision for a transaction.

        Returns
        -------
        "APPROVE" | "REVIEW" | "BLOCK"
        """
        if probability >= self.block_threshold:
            return "BLOCK"
        if probability >= self.approve_threshold:
            return "REVIEW"
        return "APPROVE"

    @staticmethod
    def get_risk_level(probability: float) -> str:
        """
        Return a human‑readable risk label independent of the decision.

        Returns
        -------
        "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        """
        if probability < 0.20:
            return "LOW"
        if probability < 0.50:
            return "MEDIUM"
        if probability < 0.80:
            return "HIGH"
        return "CRITICAL"

    def evaluate(self, probability: float) -> tuple[str, str]:
        """
        Convenience method — returns (decision, risk_level) in one call.
        """
        return self.get_decision(probability), self.get_risk_level(probability)

    # ── Optional binary flag (for UI / metrics, NOT for routing) ────────────

    def get_is_fraud(self, probability: float) -> bool:
        """
        Return a binary flag based on the model threshold (if provided),
        otherwise fall back to decision == "BLOCK".

        This method should NOT be used to alter the decision flow – it exists
        only for:
          * A/B testing different model thresholds
          * Internal dashboards (show "predicted fraud")
          * Offline evaluation

        The live routing decisions are determined by get_decision() only.
        """
        if self.model_threshold is not None:
            return probability >= self.model_threshold
        # fallback: BLOCK means fraud
        return self.get_decision(probability) == "BLOCK"
