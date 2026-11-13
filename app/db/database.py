"""Database connection and lifecycle manager.
Supports SQLite by default and MySQL if configured in DATABASE_URL.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.config import BASE_DIR, DATABASE_URL, DATA_DIR


def get_db_path() -> str:
    """Extract SQLite path from connection string or return default."""
    if DATABASE_URL.startswith("sqlite:///"):
        configured_path = Path(DATABASE_URL.removeprefix("sqlite:///"))
        if not configured_path.is_absolute():
            configured_path = BASE_DIR / configured_path
        return str(configured_path)
    return str(DATA_DIR / "smart_grid.db")


def get_connection():
    """Get active database connection with timeout and WAL mode."""
    db_path = get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=20.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables with WAL mode for concurrency."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    # Datasets metadata table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        column_count INTEGER NOT NULL,
        columns_json TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Trained models metadata table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trained_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        target TEXT NOT NULL,
        model_type TEXT NOT NULL,
        mae REAL NOT NULL,
        rmse REAL NOT NULL,
        r2 REAL NOT NULL,
        features_json TEXT NOT NULL,
        model_path TEXT NOT NULL,
        trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Prediction history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target TEXT NOT NULL,
        horizon TEXT NOT NULL,
        model_type TEXT NOT NULL,
        summary_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()


# Database access helpers
def record_dataset(filename: str, filepath: str, row_count: int, column_count: int, columns: List[str]) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO datasets (filename, filepath, row_count, column_count, columns_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (filename, filepath, row_count, column_count, json.dumps(columns))
    )
    dataset_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return dataset_id


def record_trained_model(
    model_name: str,
    target: str,
    model_type: str,
    mae: float,
    rmse: float,
    r2: float,
    features: List[str],
    model_path: str,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO trained_models (model_name, target, model_type, mae, rmse, r2, features_json, model_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (model_name, target, model_type, float(mae), float(rmse), float(r2), json.dumps(features), model_path)
    )
    model_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return model_id


def get_latest_model(target: Optional[str] = None, model_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM trained_models WHERE 1=1"
    params = []
    if target:
        query += " AND target = ?"
        params.append(target)
    if model_type:
        query += " AND model_type = ?"
        params.append(model_type)
    query += " ORDER BY id DESC LIMIT 1"
    
    cursor.execute(query, params)
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["features"] = json.loads(d["features_json"])
        return d
    return None


def get_all_models_metrics() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trained_models ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["features"] = json.loads(d["features_json"])
        result.append(d)
    return result


def record_prediction(target: str, horizon: str, model_type: str, summary: Dict[str, Any]) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO prediction_history (target, horizon, model_type, summary_json)
        VALUES (?, ?, ?, ?)
        """,
        (target, horizon, model_type, json.dumps(summary))
    )
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pred_id
