import os
import pytest
from fraud_detection.database.postgres_db import get_connection, return_connection

def test_db_connection():
    """Test that we can get and return a connection."""
    conn = get_connection()
    assert conn is not None
    cur = conn.cursor()
    cur.execute("SELECT 1")
    assert cur.fetchone()[0] == 1
    return_connection(conn)
