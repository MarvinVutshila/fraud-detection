# fraud_detection/database/postgres_db.py
from __future__ import annotations
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional

logger = logging.getLogger(__name__)

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS transactions (
    id              SERIAL PRIMARY KEY,
    transaction_id  TEXT,
    amount          REAL NOT NULL,
    probability     REAL NOT NULL,
    decision        TEXT NOT NULL,
    risk_level      TEXT NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_txn_timestamp ON transactions (timestamp DESC);
"""

CREATE_OVERRIDES_TABLE = """
CREATE TABLE IF NOT EXISTS transaction_overrides (
    transaction_id      TEXT PRIMARY KEY,
    original_decision   TEXT NOT NULL,
    new_decision        TEXT NOT NULL,
    overridden_by       TEXT NOT NULL,
    reason              TEXT,
    timestamp           TIMESTAMPTZ DEFAULT NOW()
);
"""

class Database:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._init_db()
        self._init_overrides_table()

    def _init_db(self) -> None:
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE)
            conn.commit()
        logger.info("Database initialised")

    def _init_overrides_table(self) -> None:
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_OVERRIDES_TABLE)
            conn.commit()
        logger.info("Transaction overrides table ready")

    def insert_transaction(self, transaction_id, amount, probability,
                           decision, risk_level, timestamp):
        sql = """INSERT INTO transactions (transaction_id, amount, probability,
                  decision, risk_level, timestamp)
                 VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;"""
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (transaction_id, amount, probability,
                                  decision, risk_level, timestamp))
                row_id = cur.fetchone()[0]
            conn.commit()
        return row_id

    def fetch_history(self, limit, offset, decision_filter=None):
        if decision_filter:
            sql = "SELECT * FROM transactions WHERE decision = %s ORDER BY timestamp DESC LIMIT %s OFFSET %s;"
            params = (decision_filter.upper(), limit, offset)
        else:
            sql = "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT %s OFFSET %s;"
            params = (limit, offset)
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def count_transactions(self, decision_filter=None):
        if decision_filter:
            sql = "SELECT COUNT(*) FROM transactions WHERE decision = %s;"
            params = (decision_filter.upper(),)
        else:
            sql = "SELECT COUNT(*) FROM transactions;"
            params = ()
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()[0]

    # === New methods for frontend and overrides ===

    def get_transactions(self, limit: int = 100, offset: int = 0, decision: Optional[str] = None) -> List[dict]:
        """
        Fetch recent transactions with optional decision filter.
        Returns a list of dictionaries.
        """
        if decision:
            sql = "SELECT * FROM transactions WHERE decision = %s ORDER BY timestamp DESC LIMIT %s OFFSET %s;"
            params = (decision.upper(), limit, offset)
        else:
            sql = "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT %s OFFSET %s;"
            params = (limit, offset)
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def get_transaction(self, transaction_id: str) -> Optional[dict]:
        """
        Fetch a single transaction by its transaction_id.
        Returns a dictionary or None.
        """
        sql = "SELECT * FROM transactions WHERE transaction_id = %s;"
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (transaction_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def get_override(self, transaction_id: str) -> Optional[dict]:
        """
        Fetch the override record for a given transaction.
        Returns a dictionary or None.
        """
        sql = "SELECT * FROM transaction_overrides WHERE transaction_id = %s;"
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (transaction_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def set_override(self, transaction_id: str, original_decision: str, new_decision: str,
                     overridden_by: str, reason: str) -> None:
        """
        Insert or update an override for a transaction.
        """
        sql = """
            INSERT INTO transaction_overrides (transaction_id, original_decision, new_decision, overridden_by, reason)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (transaction_id) DO UPDATE SET
                original_decision = EXCLUDED.original_decision,
                new_decision = EXCLUDED.new_decision,
                overridden_by = EXCLUDED.overridden_by,
                reason = EXCLUDED.reason,
                timestamp = NOW();
        """
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (transaction_id, original_decision, new_decision, overridden_by, reason))
            conn.commit()
