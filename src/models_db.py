#!/usr/bin/env python3
"""
Task 5: Database Trail Module
Project: 23L-0913_A2_Ops (VisionGuard)
Author: Bissam Iftikhar (23L-0913)

Persistent queryable audit trail for every inference request:
Answers: What was decided? When? By which model? How confidently?

Works seamlessly with SQLite (default local file `visionguard_history.db`)
and PostgreSQL (production Docker Compose service `postgresql://...`).
"""

import os
import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import sqlite3

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///visionguard_history.db")


def get_db_connection():
    """Returns database connection based on DATABASE_URL."""
    if DATABASE_URL.startswith("sqlite:///"):
        db_file = DATABASE_URL.replace("sqlite:///", "")
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        return conn
    elif DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"):
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
        # Fallback to local SQLite
        conn = sqlite3.connect("visionguard_history.db")
        conn.row_factory = sqlite3.Row
        return conn


def init_db():
    """Creates the inference_audit_log table if it does not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS inference_audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        prediction_label TEXT NOT NULL,
        confidence REAL NOT NULL,
        decision_threshold REAL NOT NULL,
        raw_score REAL NOT NULL,
        model_version TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        timestamp_utc TEXT NOT NULL,
        client_ip TEXT
    );
    """
    if "postgres" in DATABASE_URL:
        create_table_sql = create_table_sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")

    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()
    print("[INFO] Initialized database audit table (Task 5).")


def log_prediction(
    filename: str,
    prediction_label: str,
    confidence: float,
    decision_threshold: float,
    raw_score: float,
    model_version: str,
    latency_ms: float,
    client_ip: str = "127.0.0.1",
) -> int:
    """Inserts a new inference decision record into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if "postgres" in DATABASE_URL:
        cursor.execute(
            """
            INSERT INTO inference_audit_log 
            (filename, prediction_label, confidence, decision_threshold, raw_score, model_version, latency_ms, timestamp_utc, client_ip)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (filename, prediction_label, confidence, decision_threshold, raw_score, model_version, latency_ms, now_str, client_ip)
        )
        rec_id = cursor.fetchone()[0]
    else:
        cursor.execute(
            """
            INSERT INTO inference_audit_log 
            (filename, prediction_label, confidence, decision_threshold, raw_score, model_version, latency_ms, timestamp_utc, client_ip)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (filename, prediction_label, confidence, decision_threshold, raw_score, model_version, latency_ms, now_str, client_ip)
        )
        rec_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return rec_id


def get_all_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetches recent prediction records in reverse chronological order."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM inference_audit_log ORDER BY id DESC LIMIT ?"
    if "postgres" in DATABASE_URL:
        query = "SELECT * FROM inference_audit_log ORDER BY id DESC LIMIT %s"
        cursor.execute(query, (limit,))
        desc = [d[0] for d in cursor.description]
        records = [dict(zip(desc, row)) for row in cursor.fetchall()]
    else:
        cursor.execute(query, (limit,))
        records = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return records


if __name__ == "__main__":
    init_db()
    test_id = log_prediction(
        filename="test_sample.jpg",
        prediction_label="Real Image",
        confidence=94.50,
        decision_threshold=0.50,
        raw_score=0.0550,
        model_version="VisionGuard-MobileNetV2-v2@champion",
        latency_ms=18.4,
    )
    print(f"[SUCCESS] Logged test prediction with record ID #{test_id}!")
    print(f"Current history length: {len(get_all_logs())} entries.")
