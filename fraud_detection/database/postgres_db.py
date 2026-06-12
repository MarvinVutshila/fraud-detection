from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import List, Optional, Dict, Any

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

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

CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    username        TEXT UNIQUE NOT NULL,
    password        TEXT NOT NULL,
    role            TEXT DEFAULT 'analyst',
    status          TEXT DEFAULT 'pending',
    avatar_url      TEXT,
    failed_attempts INTEGER DEFAULT 0,
    lock_until      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_LOGIN_LOGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS login_logs (
    id          SERIAL PRIMARY KEY,
    username    TEXT,
    success     BOOLEAN,
    ip          TEXT,
    user_agent  TEXT,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_USER_ACTIVITY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_activity (
    id          SERIAL PRIMARY KEY,
    username    TEXT,
    action      TEXT,
    details     JSONB,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);
"""

# -------------------------------------------------------------------
# Global connection pool (initialised once at startup)
# -------------------------------------------------------------------
_pool: Optional[SimpleConnectionPool] = None
_dsn: Optional[str] = None


def init_db_pool(dsn: str, min_conn: int = 1, max_conn: int = 20) -> None:
    """Call this ONCE when your application starts."""
    global _pool, _dsn
    _dsn = dsn
    _pool = SimpleConnectionPool(min_conn, max_conn, dsn=dsn)
    logger.info(f"Database pool created (min={min_conn}, max={max_conn})")


def create_tables() -> None:
    """Run table creation – call once at startup after init_db_pool()."""
    if _pool is None:
        raise RuntimeError("Database pool not initialised. Call init_db_pool() first.")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(CREATE_OVERRIDES_TABLE_SQL)
            cur.execute(CREATE_USERS_TABLE_SQL)
            cur.execute(CREATE_LOGIN_LOGS_TABLE_SQL)
            cur.execute(CREATE_USER_ACTIVITY_TABLE_SQL)
        conn.commit()
    logger.info("Database tables and indexes verified")


@contextmanager
def get_connection():
    """
    Context manager that returns a WORKING connection from the pool.
    Tests the connection with a simple query; if dead, discards it and retries.
    This solves the "SSL connection has been closed unexpectedly" error.
    """
    if _pool is None:
        raise RuntimeError("Database pool not initialised. Call init_db_pool() first.")

    conn = None
    max_attempts = 3
    for attempt in range(max_attempts):
        conn = _pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            break
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            logger.warning(f"Bad connection (attempt {attempt+1}/{max_attempts}): {e}")
            conn.close()
            conn = None
            if attempt == max_attempts - 1:
                raise RuntimeError("Could not get a working database connection") from e
            continue

    try:
        yield conn
    finally:
        if conn is not None:
            _pool.putconn(conn)


# -------------------------------------------------------------------
# Database class – uses the pool internally, no per-query overhead
# -------------------------------------------------------------------
class Database:
    """
    Main database accessor. Reuses the global connection pool.
    Do NOT create a new instance per request – create ONE global instance.
    """

    def __init__(self) -> None:
        if _pool is None:
            raise RuntimeError("Database pool not initialised. Call init_db_pool() first.")

    def insert_transaction(self, transaction_id: str, amount: float,
                           probability: float, decision: str,
                           risk_level: str, timestamp) -> int:
        sql = """
            INSERT INTO transactions (transaction_id, amount, probability,
                                      decision, risk_level, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (transaction_id, amount, probability,
                                  decision, risk_level, timestamp))
                row_id = cur.fetchone()[0]
            conn.commit()
        return row_id

    def fetch_history(self, limit: int, offset: int,
                      decision_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if decision_filter:
            sql = """
                SELECT * FROM transactions
                WHERE decision = %s
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s;
            """
            params = (decision_filter.upper(), limit, offset)
        else:
            sql = """
                SELECT * FROM transactions
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s;
            """
            params = (limit, offset)

        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def count_transactions(self, decision_filter: Optional[str] = None) -> int:
        if decision_filter:
            sql = "SELECT COUNT(*) FROM transactions WHERE decision = %s;"
            params = (decision_filter.upper(),)
        else:
            sql = "SELECT COUNT(*) FROM transactions;"
            params = ()

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()[0]

    def get_transactions(self, limit: int = 100, offset: int = 0,
                         decision: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.fetch_history(limit, offset, decision)

    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM transactions WHERE transaction_id = %s;"
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (transaction_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def get_override(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM transaction_overrides WHERE transaction_id = %s;"
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (transaction_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def set_override(self, transaction_id: str, original_decision: str,
                     new_decision: str, overridden_by: str, reason: str) -> None:
        sql = """
            INSERT INTO transaction_overrides
                (transaction_id, original_decision, new_decision, overridden_by, reason)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (transaction_id) DO UPDATE SET
                original_decision = EXCLUDED.original_decision,
                new_decision = EXCLUDED.new_decision,
                overridden_by = EXCLUDED.overridden_by,
                reason = EXCLUDED.reason,
                timestamp = NOW();
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (transaction_id, original_decision, new_decision,
                                  overridden_by, reason))
            conn.commit()

    # ✅ NEW METHOD: Update the main transaction decision after override
    def update_transaction_decision(self, transaction_id: str, new_decision: str) -> None:
        sql = "UPDATE transactions SET decision = %s WHERE transaction_id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (new_decision, transaction_id))
            conn.commit()
