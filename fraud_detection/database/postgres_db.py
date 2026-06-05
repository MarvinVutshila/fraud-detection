# database/postgres_db.py
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

class Database:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._init_db()

    def _init_db(self) -> None:
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE)
            conn.commit()
        logger.info("Database initialised")

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