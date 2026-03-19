"""
SQLite Memory Store — sessions, papers, insights, artifacts, metrics, conversation.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

logger  = logging.getLogger(__name__)
DB_PATH = "data/research_memory.db"


class MemoryStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"[MemoryStore] Connected to {self.db_path}")

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, query TEXT NOT NULL,
                subtopics TEXT, filters TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, paper_id TEXT NOT NULL,
                title TEXT, authors TEXT, abstract TEXT, published TEXT,
                arxiv_url TEXT, categories TEXT, citation_count INTEGER DEFAULT 0,
                venue TEXT, final_score REAL DEFAULT 0.0, score_breakdown TEXT,
                UNIQUE(session_id, paper_id)
            );
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, paper_id TEXT NOT NULL,
                insight_type TEXT, content TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, artifact_type TEXT,
                content TEXT, file_path TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, query_time_sec REAL,
                papers_retrieved INTEGER, papers_selected INTEGER,
                score_mean REAL, score_std REAL, user_rating INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, role TEXT,
                message TEXT, created_at TEXT NOT NULL
            );
        """)
        self.conn.commit()

    # --- Sessions ---
    def create_session(self, session_id: str, query: str, subtopics: list = None):
        now = datetime.utcnow().isoformat()
        self.conn.execute("""
            INSERT INTO sessions (session_id,query,subtopics,filters,created_at,updated_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(session_id) DO UPDATE SET query=excluded.query,
            subtopics=excluded.subtopics, updated_at=excluded.updated_at
        """, (session_id, query, json.dumps(subtopics or []), json.dumps({}), now, now))
        self.conn.commit()

    def get_session(self, session_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_session_filters(self, session_id: str, filters: dict):
        self.conn.execute(
            "UPDATE sessions SET filters=?,updated_at=? WHERE session_id=?",
            (json.dumps(filters), datetime.utcnow().isoformat(), session_id)
        )
        self.conn.commit()

    # --- Papers ---
    def save_papers(self, session_id: str, papers: list[dict]):
        for p in papers:
            self.conn.execute("""
                INSERT OR REPLACE INTO papers
                (session_id,paper_id,title,authors,abstract,published,arxiv_url,
                 categories,citation_count,venue,final_score,score_breakdown)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (session_id, p.get("paper_id",""), p.get("title",""),
                  json.dumps(p.get("authors",[])), p.get("abstract",""),
                  p.get("published",""), p.get("arxiv_url",""),
                  json.dumps(p.get("categories",[])), p.get("citation_count",0),
                  p.get("venue",""), p.get("final_score",0.0),
                  json.dumps(p.get("score_breakdown",{}))))
        self.conn.commit()

    def get_papers(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE session_id=? ORDER BY final_score DESC", (session_id,)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["authors"]        = json.loads(d["authors"] or "[]")
            d["categories"]     = json.loads(d["categories"] or "[]")
            d["score_breakdown"] = json.loads(d["score_breakdown"] or "{}")
            results.append(d)
        return results

    # --- Insights ---
    def save_insight(self, session_id: str, paper_id: str, insight_type: str, content: str):
        self.conn.execute(
            "INSERT INTO insights (session_id,paper_id,insight_type,content,created_at) VALUES (?,?,?,?,?)",
            (session_id, paper_id, insight_type, content, datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def get_insights(self, session_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM insights WHERE session_id=?", (session_id,)
        ).fetchall()]

    # --- Artifacts ---
    def save_artifact(self, session_id: str, artifact_type: str, content: str, file_path: str = ""):
        self.conn.execute(
            "INSERT INTO artifacts (session_id,artifact_type,content,file_path,created_at) VALUES (?,?,?,?,?)",
            (session_id, artifact_type, content, file_path, datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def get_artifact(self, session_id: str, artifact_type: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT content FROM artifacts WHERE session_id=? AND artifact_type=? ORDER BY id DESC LIMIT 1",
            (session_id, artifact_type)
        ).fetchone()
        return row["content"] if row else None

    # --- Metrics ---
    def save_metrics(self, session_id: str, metrics: dict):
        self.conn.execute("""
            INSERT INTO metrics
            (session_id,query_time_sec,papers_retrieved,papers_selected,
             score_mean,score_std,user_rating,created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (session_id, metrics.get("query_time_sec",0), metrics.get("papers_retrieved",0),
              metrics.get("papers_selected",0), metrics.get("score_mean",0),
              metrics.get("score_std",0), metrics.get("user_rating"),
              datetime.utcnow().isoformat()))
        self.conn.commit()

    def get_metrics(self, session_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM metrics WHERE session_id=?", (session_id,)
        ).fetchall()]

    # --- Conversation ---
    def add_message(self, session_id: str, role: str, message: str):
        self.conn.execute(
            "INSERT INTO conversation (session_id,role,message,created_at) VALUES (?,?,?,?)",
            (session_id, role, message, datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def get_conversation(self, session_id: str, last_n: int = 20) -> list[dict]:
        rows = self.conn.execute("""
            SELECT role,message,created_at FROM conversation
            WHERE session_id=? ORDER BY id DESC LIMIT ?
        """, (session_id, last_n)).fetchall()
        return list(reversed([dict(r) for r in rows]))

    # --- Utility ---
    def list_sessions(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT session_id,query,created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()]

    def close(self):
        self.conn.close()