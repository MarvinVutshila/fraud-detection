# services/storage_service.py
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import List, Optional
from fraud_detection.database.postgres_db import Database
from fraud_detection.models.schemas import HistoryRecord

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self, db: Database) -> None:
        self._db = db

    def store(self, transaction_id, amount, probability, decision, risk_level):
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        try:
            row_id = self._db.insert_transaction(
                transaction_id=transaction_id,
                amount=amount,
                probability=probability,
                decision=decision,
                risk_level=risk_level,
                timestamp=timestamp,
            )
            logger.debug("Stored row %d", row_id)
        except Exception as exc:
            logger.error("Failed to store transaction %s: %s", transaction_id, exc)

    def get_history(self, limit=100, offset=0, decision_filter=None):
        raw = self._db.fetch_history(limit, offset, decision_filter)
        total = self._db.count_transactions(decision_filter)
        records = [HistoryRecord(**r) for r in raw]
        return records, total