from __future__ import annotations

import logging

import numpy as np
from fastapi import HTTPException

from fraud_detection.core.config import MAX_KNOWN_AMOUNT, SHAP_TOP_N
from fraud_detection.models.model_loader import ModelArtefacts
from fraud_detection.models.schemas import (
    ExplanationOutput,
    PredictionResponse,
    TransactionRequest,
)
from fraud_detection.services.decision_service import DecisionService
from fraud_detection.services.storage_service import StorageService
from fraud_detection.utils.feature_engineering import engineer_features
from fraud_detection.utils.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(
        self,
        artefacts: ModelArtefacts,
        decision_service: DecisionService,
        storage_service: StorageService,
    ) -> None:
        self._artefacts = artefacts
        self._decision_service = decision_service
        self._storage_service = storage_service
        self._shap = SHAPExplainer(
            model=artefacts.model,
            feature_names=artefacts.feature_names,
        )
        logger.info("PredictionService ready")

    def predict(
        self,
        tx: TransactionRequest,
        explain: bool = True,
    ) -> PredictionResponse:
        # Amount guard
        self._validate_amount(tx)

        # Feature engineering
        X_raw = engineer_features(
            tx_dict=tx.model_dump(),
            feature_names=self._artefacts.feature_names,
            amount_bins=self._artefacts.amount_bins,
        )

        # Scaling
        X_scaled = self._artefacts.scaler.transform(X_raw)

        # Inference
        prob = float(self._artefacts.model.predict_proba(X_scaled)[0, 1])

        # Decision
        decision, risk_level = self._decision_service.evaluate(prob)

        # SHAP
        explanation: ExplanationOutput | None = None
        if explain:
            top_features, contributions = self._shap.explain(
                X_scaled, top_n=SHAP_TOP_N
            )
            if top_features:
                explanation = ExplanationOutput(
                    top_features=top_features,
                    feature_contributions=contributions,
                )

        logger.info(
            "Prediction | tx=%s  prob=%.4f  decision=%s  risk=%s",
            tx.transaction_id, prob, decision, risk_level,
        )

        # Persist
        self._storage_service.store(
            transaction_id=tx.transaction_id,
            amount=tx.Amount,
            probability=prob,
            decision=decision,
            risk_level=risk_level,
        )

        return PredictionResponse(
            transaction_id=tx.transaction_id,
            fraud_probability=round(prob, 6),
            decision=decision,
            risk_level=risk_level,
            threshold=self._artefacts.optimal_threshold,
            explanation=explanation,
            is_fraud=(decision == "BLOCK")   # <-- FIXED: added required field
        )

    @staticmethod
    def _validate_amount(tx: TransactionRequest) -> None:
        if tx.Amount > MAX_KNOWN_AMOUNT:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Transaction amount ${tx.Amount:,.2f} exceeds the maximum "
                    f"allowed amount ${MAX_KNOWN_AMOUNT:,.2f}. "
                    "Contact support if this is a legitimate transaction."
                ),
            )

    def model_info(self) -> dict:
        a = self._artefacts
        return {
            "model_type": type(a.model).__name__,
            "n_features": len(a.feature_names),
            "feature_names": a.feature_names,
            "optimal_threshold": a.optimal_threshold,
            "max_allowed_amount": MAX_KNOWN_AMOUNT,
            "approve_threshold": self._decision_service.approve_threshold,
            "block_threshold": self._decision_service.block_threshold,
        }