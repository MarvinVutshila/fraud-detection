# fraud_detection/database/postgres_db.py
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import List, Optional, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# Global engine instance (initialised once at startup)
_engine = None

# -------------------------------------------------------------------
# SQL statements (run only once)
# -------------------------------------------------------------------
CREATE_TABLE_SQL = """
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

CREATE_OVERRIDES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transaction_overrides (
    transaction_id      TEXT PRIMARY KEY,
    original_decision   TEXT NOT NULL,
    new_decision        TEXT NOT NULL,
    overridden_by       TEXT NOT NULL,
    reason              TEXT,
    timestamp           TIMESTAMPTZ DEFAULT NOW()
);
"""


def init_db_pool(dsn: str, min_conn: int = 1, max_conn: int = 20) -> None:
    """Call this ONCE when your application starts."""
    global _engine
    if _engine is not None:
        logger.warning("Database pool already initialised, ignoring call.")
        return
    _engine = create_engine(
        dsn,
        poolclass=QueuePool,
        pool_size=max_conn,
        max_overflow=0,        # no extra overflow beyond pool_size
        pool_pre_ping=True,    # 🔥 tests connection before using – kills dead ones
        pool_recycle=300,      # recycle every 5 minutes to avoid stale connections
        echo=False,
    )
    logger.info(f"Database pool created (size={max_conn})")


def create_tables() -> None:
    """Create tables if they don't exist. Call after init_db_pool()."""
    if _engine is None:
        raise RuntimeError("init_db_pool() must be called first.")
    with _engine.connect() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        conn.execute(text(CREATE_OVERRIDES_TABLE_SQL))
        conn.commit()
    logger.info("Database tables and indexes verified")


@contextmanager
def get_connection():
    """
    Context manager that yields a working SQLAlchemy connection from the pool.
    The connection is automatically returned to the pool when the block exits.
    """
    if _engine is None:
        raise RuntimeError("init_db_pool() must be called first.")
    conn = _engine.connect()
    try:
        yield conn
    finally:
        conn.close()


# -------------------------------------------------------------------
# Database class – uses the pool internally
# -------------------------------------------------------------------
class Database:
    """
    Main database accessor. Uses the global connection pool.
    Create ONE global instance after init_db_pool().
    """

    def __init__(self) -> None:
        if _engine is None:
            raise RuntimeError("init_db_pool() must be called before creating Database instance.")

    def insert_transaction(self, transaction_id: str, amount: float,
                           probability: float, decision: str,
                           risk_level: str, timestamp) -> int:
        sql = """
            INSERT INTO transactions (transaction_id, amount, probability,
                                      decision, risk_level, timestamp)
            VALUES (:transaction_id, :amount, :probability,
                    :decision, :risk_level, :timestamp)
            RETURNING id;
        """
        with get_connection() as conn:
            result = conn.execute(
                text(sql),
                {
                    "transaction_id": transaction_id,
                    "amount": amount,
                    "probability": probability,
                    "decision": decision,
                    "risk_level": risk_level,
                    "timestamp": timestamp,
                },
            )
            row_id = result.fetchone()[0]
            conn.commit()
        return row_id

    def fetch_history(self, limit: int, offset: int,
                      decision_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if decision_filter:
            sql = """
                SELECT * FROM transactions
                WHERE decision = :decision
                ORDER BY timestamp DESC
                LIMIT :limit OFFSET :offset;
            """
            params = {"decision": decision_filter.upper(), "limit": limit, "offset": offset}
        else:
            sql = """
                SELECT * FROM transactions
                ORDER BY timestamp DESC
                LIMIT :limit OFFSET :offset;
            """
            params = {"limit": limit, "offset": offset}

        with get_connection() as conn:
            result = conn.execute(text(sql), params)
            rows = result.mappings().all()
        return [dict(row) for row in rows]

    def count_transactions(self, decision_filter: Optional[str] = None) -> int:
        if decision_filter:
            sql = "SELECT COUNT(*) FROM transactions WHERE decision = :decision;"
            params = {"decision": decision_filter.upper()}
        else:
            sql = "SELECT COUNT(*) FROM transactions;"
            params = {}
        with get_connection() as conn:
            result = conn.execute(text(sql), params)
            return result.scalar()

    def get_transactions(self, limit: int = 100, offset: int = 0,
                         decision: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.fetch_history(limit, offset, decision)

    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM transactions WHERE transaction_id = :tx_id;"
        with get_connection() as conn:
            result = conn.execute(text(sql), {"tx_id": transaction_id})
            row = result.mappings().first()
        return dict(row) if row else None

    def get_override(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM transaction_overrides WHERE transaction_id = :tx_id;"
        with get_connection() as conn:
            result = conn.execute(text(sql), {"tx_id": transaction_id})
            row = result.mappings().first()
        return dict(row) if row else None

    def set_override(self, transaction_id: str, original_decision: str,
                     new_decision: str, overridden_by: str, reason: str) -> None:
        sql = """
            INSERT INTO transaction_overrides
                (transaction_id, original_decision, new_decision, overridden_by, reason)
            VALUES (:tx_id, :orig, :new, :by, :reason)
            ON CONFLICT (transaction_id) DO UPDATE SET
                original_decision = EXCLUDED.original_decision,
                new_decision = EXCLUDED.new_decision,
                overridden_by = EXCLUDED.overridden_by,
                reason = EXCLUDED.reason,
                timestamp = NOW();
        """
        with get_connection() as conn:
            conn.execute(
                text(sql),
                {
                    "tx_id": transaction_id,
                    "orig": original_decision,
                    "new": new_decision,
                    "by": overridden_by,
                    "reason": reason,
                },
            )
            conn.commit()
