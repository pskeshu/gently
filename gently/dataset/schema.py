"""
SQLite database schema for embryo dataset.

Tables:
- sessions: Session metadata
- embryos: Embryo records linked to sessions
- volumes: 3D volume data with file paths
- images: 2D projections linked to volumes
- ground_truth: Human-annotated stage labels
- perception_runs: Test run configurations
- predictions: Per-image stage predictions
- observed_features: Structured features from perception
- reasoning_traces: Full VLM reasoning traces
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DATABASE_VERSION = 1
DEFAULT_DB_PATH = Path("D:/gently/dataset.db")


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get a database connection with optimized settings.

    Parameters
    ----------
    db_path : Path, optional
        Path to database file. Defaults to D:/gently/dataset.db

    Returns
    -------
    sqlite3.Connection
        Database connection with row factory set
    """
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Context manager for database transactions."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_database(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Initialize the database with all tables.

    Parameters
    ----------
    db_path : Path, optional
        Path to database file. Creates parent directories if needed.

    Returns
    -------
    sqlite3.Connection
        Initialized database connection
    """
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(path)

    # Create tables
    conn.executescript(SCHEMA_SQL)

    # Set version
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('version', ?)",
        (str(DATABASE_VERSION),)
    )
    conn.commit()

    logger.info(f"Database initialized at {path}")
    return conn


SCHEMA_SQL = """
-- Metadata table for tracking database version
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    created_at TIMESTAMP NOT NULL,
    last_active TIMESTAMP,
    metadata_json TEXT  -- Additional JSON metadata
);

-- Embryos table
CREATE TABLE IF NOT EXISTS embryos (
    embryo_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    nickname TEXT,
    user_label TEXT,
    stage_position_x REAL,
    stage_position_y REAL,
    calibration_json TEXT,  -- Full calibration data as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (embryo_id, session_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_embryos_session ON embryos(session_id);

-- Volumes table (3D TIFF data)
CREATE TABLE IF NOT EXISTS volumes (
    uid TEXT PRIMARY KEY,
    session_id TEXT,
    embryo_id TEXT,
    timepoint INTEGER,
    file_path TEXT NOT NULL,  -- Path to TIFF file
    shape_json TEXT,  -- Array dimensions as JSON [views, z, y, x]
    dtype TEXT,
    timestamp TIMESTAMP NOT NULL,
    metadata_json TEXT,  -- Additional metadata
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_volumes_session ON volumes(session_id);
CREATE INDEX IF NOT EXISTS idx_volumes_embryo ON volumes(embryo_id, timepoint);
CREATE INDEX IF NOT EXISTS idx_volumes_timestamp ON volumes(timestamp);

-- Images table (2D projections)
CREATE TABLE IF NOT EXISTS images (
    uid TEXT PRIMARY KEY,
    parent_uid TEXT,  -- Link to source volume
    session_id TEXT,
    embryo_id TEXT,
    timepoint INTEGER,
    projection_type TEXT,  -- 'max_z', 'mean_z', etc.
    file_path TEXT,  -- Path to image file if stored separately
    shape_json TEXT,  -- [height, width]
    dtype TEXT,
    b64_size_kb REAL,  -- Size when base64 encoded
    timestamp TIMESTAMP NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (parent_uid) REFERENCES volumes(uid),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_images_volume ON images(parent_uid);
CREATE INDEX IF NOT EXISTS idx_images_embryo ON images(embryo_id, timepoint);
CREATE INDEX IF NOT EXISTS idx_images_timestamp ON images(timestamp);

-- Ground truth table (human annotations)
CREATE TABLE IF NOT EXISTS ground_truth (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    embryo_id TEXT NOT NULL,
    stage TEXT NOT NULL,  -- early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched
    start_timepoint INTEGER NOT NULL,  -- Timepoint when this stage starts
    annotator TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (session_id, embryo_id, stage),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_gt_session ON ground_truth(session_id);
CREATE INDEX IF NOT EXISTS idx_gt_embryo ON ground_truth(session_id, embryo_id);

-- Perception runs table (test configurations)
CREATE TABLE IF NOT EXISTS perception_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    perception_method TEXT NOT NULL,  -- 'vlm_v1', 'vlm_v2', 'vlm_v3', etc.
    model_name TEXT,  -- 'claude-sonnet-4-5-20250929', etc.
    config_json TEXT,  -- Full configuration as JSON
    filter_criteria_json TEXT,  -- What data was selected
    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
    total_samples INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON perception_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_method ON perception_runs(perception_method);

-- Predictions table (per-image results)
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    perception_run_id INTEGER NOT NULL,
    image_uid TEXT,
    volume_uid TEXT,
    session_id TEXT,
    embryo_id TEXT,
    timepoint INTEGER,

    -- Prediction results
    predicted_stage TEXT NOT NULL,
    confidence REAL,  -- 0.0-1.0
    confidence_level TEXT,  -- 'HIGH', 'MEDIUM', 'LOW'
    is_transitional INTEGER DEFAULT 0,
    transition_from TEXT,
    transition_to TEXT,
    reasoning TEXT,  -- Summary reasoning

    -- Comparison with ground truth (computed)
    ground_truth_stage TEXT,
    is_correct INTEGER,  -- 1 if matches ground truth, 0 otherwise, NULL if no GT

    -- Metadata
    perception_version TEXT,  -- 'v1', 'v2', 'v3' for format versioning
    execution_time_ms REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (perception_run_id) REFERENCES perception_runs(id),
    FOREIGN KEY (image_uid) REFERENCES images(uid),
    FOREIGN KEY (volume_uid) REFERENCES volumes(uid)
);
CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions(perception_run_id);
CREATE INDEX IF NOT EXISTS idx_pred_image ON predictions(image_uid);
CREATE INDEX IF NOT EXISTS idx_pred_embryo ON predictions(embryo_id, timepoint);
CREATE INDEX IF NOT EXISTS idx_pred_stage ON predictions(predicted_stage);

-- Observed features table (structured features from v3 perception)
CREATE TABLE IF NOT EXISTS observed_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,

    -- Feature fields
    shape TEXT,  -- 'oval', 'elongated', 'comma', etc.
    curvature TEXT,  -- 'none', 'slight', 'pronounced'
    shell_status TEXT,  -- 'intact', 'breached', 'absent'
    body_segments TEXT,  -- Description of visible segments
    emergence TEXT,  -- 'none', 'partial', 'complete'
    movement TEXT,  -- 'none', 'slight', 'active'
    texture TEXT,  -- 'smooth', 'granular'

    -- Raw JSON for extensibility
    features_json TEXT,

    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);
CREATE INDEX IF NOT EXISTS idx_features_pred ON observed_features(prediction_id);

-- Reasoning traces table (full VLM traces for v3)
CREATE TABLE IF NOT EXISTS reasoning_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,

    -- Trace data
    contrastive_reasoning TEXT,  -- Why not adjacent stages
    steps_json TEXT,  -- List of reasoning steps as JSON
    tool_calls_json TEXT,  -- Tool calls made
    tools_used_json TEXT,  -- Names of tools used
    total_tool_calls INTEGER,

    -- Token usage
    input_tokens INTEGER,
    output_tokens INTEGER,

    -- Verification phase data (if triggered)
    verification_triggered INTEGER DEFAULT 0,
    verification_result_json TEXT,

    -- Raw response for debugging
    raw_response TEXT,

    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);
CREATE INDEX IF NOT EXISTS idx_traces_pred ON reasoning_traces(prediction_id);

-- Aggregation tracking table
CREATE TABLE IF NOT EXISTS aggregation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,  -- 'sessions', 'volumes', 'images', 'ground_truth'
    source_path TEXT,
    items_processed INTEGER,
    items_added INTEGER,
    items_updated INTEGER,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status TEXT DEFAULT 'running',  -- running, completed, failed
    error_message TEXT
);

-- View: Get stage at timepoint (using ground truth)
CREATE VIEW IF NOT EXISTS v_stage_at_timepoint AS
SELECT
    gt.session_id,
    gt.embryo_id,
    gt.stage,
    gt.start_timepoint,
    COALESCE(
        (SELECT MIN(gt2.start_timepoint) - 1
         FROM ground_truth gt2
         WHERE gt2.session_id = gt.session_id
           AND gt2.embryo_id = gt.embryo_id
           AND gt2.start_timepoint > gt.start_timepoint),
        999999
    ) as end_timepoint
FROM ground_truth gt;

-- View: Prediction accuracy summary by run
CREATE VIEW IF NOT EXISTS v_run_accuracy AS
SELECT
    pr.id as run_id,
    pr.name,
    pr.perception_method,
    pr.model_name,
    COUNT(p.id) as total_predictions,
    SUM(CASE WHEN p.is_correct = 1 THEN 1 ELSE 0 END) as correct,
    SUM(CASE WHEN p.is_correct = 0 THEN 1 ELSE 0 END) as incorrect,
    SUM(CASE WHEN p.is_correct IS NULL THEN 1 ELSE 0 END) as no_ground_truth,
    ROUND(100.0 * SUM(CASE WHEN p.is_correct = 1 THEN 1 ELSE 0 END) /
          NULLIF(SUM(CASE WHEN p.is_correct IS NOT NULL THEN 1 ELSE 0 END), 0), 2) as accuracy_pct
FROM perception_runs pr
LEFT JOIN predictions p ON pr.id = p.perception_run_id
GROUP BY pr.id;
"""


def get_database_stats(conn: sqlite3.Connection) -> dict:
    """
    Get statistics about the database contents.

    Returns
    -------
    dict
        Statistics including counts of sessions, embryos, volumes, etc.
    """
    stats = {}

    tables = ['sessions', 'embryos', 'volumes', 'images', 'ground_truth',
              'perception_runs', 'predictions']

    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        stats[table] = count

    # Additional stats
    stats['unique_embryo_sessions'] = conn.execute(
        "SELECT COUNT(DISTINCT session_id || embryo_id) FROM volumes"
    ).fetchone()[0]

    # Date range
    result = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM volumes"
    ).fetchone()
    stats['earliest_volume'] = result[0]
    stats['latest_volume'] = result[1]

    return stats
