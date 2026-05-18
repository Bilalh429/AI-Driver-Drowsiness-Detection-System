"""
utils/database.py
=================
SQLite-based detection log storage + PDF session report generator.

Database schema
---------------
Table: detection_events
  id              INTEGER PRIMARY KEY AUTOINCREMENT
  timestamp       TEXT     ISO-8601
  session_id      TEXT     UUID for grouping events per run
  ear             REAL
  mar             REAL
  eyes_closed     INTEGER  0/1
  yawning         INTEGER  0/1
  drowsy          INTEGER  0/1
  alarm_triggered INTEGER  0/1
  fps             REAL

Table: sessions
  session_id   TEXT PRIMARY KEY
  started_at   TEXT
  ended_at     TEXT
  total_frames INTEGER
  drowsy_events INTEGER
  yawn_events   INTEGER
  alarm_events  INTEGER
  avg_ear       REAL
  avg_mar       REAL
"""

import sqlite3
import uuid
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── ReportLab is optional — PDF generation gracefully disabled if absent ─────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False
    logger.warning("reportlab not installed — PDF reports disabled. "
                   "Run: pip install reportlab")


class DetectionDatabase:
    """
    Manages SQLite storage for detection events and session summaries.

    Parameters
    ----------
    db_path : str  Path to the SQLite database file.
    """

    def __init__(self, db_path: str):
        self._db_path   = db_path
        self._session_id = str(uuid.uuid4())
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._total_frames  = 0
        self._drowsy_events = 0
        self._yawn_events   = 0
        self._alarm_events  = 0
        self._ear_sum = 0.0
        self._mar_sum = 0.0

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        self._start_session()
        logger.info("DetectionDatabase ready (session %s).", self._session_id[:8])

    # ── DB setup ─────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS detection_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    session_id      TEXT NOT NULL,
                    ear             REAL,
                    mar             REAL,
                    eyes_closed     INTEGER,
                    yawning         INTEGER,
                    drowsy          INTEGER,
                    alarm_triggered INTEGER,
                    fps             REAL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id      TEXT PRIMARY KEY,
                    started_at      TEXT,
                    ended_at        TEXT,
                    total_frames    INTEGER DEFAULT 0,
                    drowsy_events   INTEGER DEFAULT 0,
                    yawn_events     INTEGER DEFAULT 0,
                    alarm_events    INTEGER DEFAULT 0,
                    avg_ear         REAL DEFAULT 0,
                    avg_mar         REAL DEFAULT 0
                );
            """)

    def _start_session(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, started_at) VALUES (?, ?)",
                (self._session_id, self._started_at),
            )

    # ── public API ────────────────────────────────────────────────────────────

    def log_event(self, ear: float, mar: float, eyes_closed: bool,
                  yawning: bool, drowsy: bool, alarm_triggered: bool,
                  fps: float) -> None:
        """Insert one detection event row."""
        self._total_frames  += 1
        self._ear_sum       += ear
        self._mar_sum       += mar
        if drowsy:          self._drowsy_events += 1
        if yawning:         self._yawn_events   += 1
        if alarm_triggered: self._alarm_events  += 1

        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO detection_events
                       (timestamp, session_id, ear, mar, eyes_closed,
                        yawning, drowsy, alarm_triggered, fps)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now().isoformat(timespec="milliseconds"),
                        self._session_id,
                        round(ear, 4), round(mar, 4),
                        int(eyes_closed), int(yawning),
                        int(drowsy), int(alarm_triggered),
                        round(fps, 1),
                    ),
                )
        except sqlite3.Error as exc:
            logger.error("DB write error: %s", exc)

    def close_session(self) -> None:
        """Finalise session statistics in the sessions table."""
        avg_ear = self._ear_sum / self._total_frames if self._total_frames else 0
        avg_mar = self._mar_sum / self._total_frames if self._total_frames else 0

        try:
            with self._connect() as conn:
                conn.execute(
                    """UPDATE sessions SET
                           ended_at      = ?,
                           total_frames  = ?,
                           drowsy_events = ?,
                           yawn_events   = ?,
                           alarm_events  = ?,
                           avg_ear       = ?,
                           avg_mar       = ?
                       WHERE session_id = ?""",
                    (
                        datetime.now().isoformat(timespec="seconds"),
                        self._total_frames,
                        self._drowsy_events,
                        self._yawn_events,
                        self._alarm_events,
                        round(avg_ear, 4),
                        round(avg_mar, 4),
                        self._session_id,
                    ),
                )
            logger.info("Session %s closed.", self._session_id[:8])
        except sqlite3.Error as exc:
            logger.error("DB session close error: %s", exc)

    def get_session_summary(self) -> dict:
        """Return aggregate stats for the current session."""
        avg_ear = self._ear_sum / self._total_frames if self._total_frames else 0
        avg_mar = self._mar_sum / self._total_frames if self._total_frames else 0
        return {
            "session_id":    self._session_id,
            "started_at":    self._started_at,
            "total_frames":  self._total_frames,
            "drowsy_events": self._drowsy_events,
            "yawn_events":   self._yawn_events,
            "alarm_events":  self._alarm_events,
            "avg_ear":       round(avg_ear, 4),
            "avg_mar":       round(avg_mar, 4),
        }

    def get_all_sessions(self) -> list[dict]:
        """Return all sessions ordered newest-first."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM sessions ORDER BY started_at DESC"
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error:
            return []

    def get_recent_events(self, limit: int = 500) -> list[dict]:
        """Return the most recent *limit* detection events."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """SELECT * FROM detection_events
                       ORDER BY id DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in reversed(rows)]
        except sqlite3.Error:
            return []

    @property
    def session_id(self) -> str:
        return self._session_id


# ── PDF Report Generator ─────────────────────────────────────────────────────

def generate_pdf_report(db: DetectionDatabase,
                        output_path: str) -> Optional[str]:
    """
    Generate a PDF session report from the current session's data.

    Parameters
    ----------
    db          : DetectionDatabase  Live database instance.
    output_path : str                Where to save the PDF.

    Returns
    -------
    str  Path to the generated PDF, or None if reportlab is missing.
    """
    if not _REPORTLAB:
        logger.error("reportlab not installed — cannot generate PDF.")
        return None

    summary = db.get_session_summary()
    events  = db.get_recent_events(limit=1000)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc    = SimpleDocTemplate(output_path, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # ── Title ────────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=10, textColor=colors.grey, spaceAfter=14,
    )
    story.append(Paragraph("🚗 Driver Drowsiness Detection — Session Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Session ID: {summary['session_id'][:16]}…",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#4a90d9")))
    story.append(Spacer(1, 0.4*cm))

    # ── Summary cards ────────────────────────────────────────────────────────
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#2c3e50"), spaceBefore=10,
    )
    story.append(Paragraph("Session Summary", section_style))

    summary_data = [
        ["Metric", "Value"],
        ["Session Started",   summary["started_at"]],
        ["Total Frames Processed", f"{summary['total_frames']:,}"],
        ["Drowsiness Events", str(summary["drowsy_events"])],
        ["Yawn Events",       str(summary["yawn_events"])],
        ["Alarm Triggers",    str(summary["alarm_events"])],
        ["Average EAR",       f"{summary['avg_ear']:.4f}"],
        ["Average MAR",       f"{summary['avg_mar']:.4f}"],
    ]

    tbl = Table(summary_data, colWidths=[9*cm, 8*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a90d9")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 11),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 1), (-1, -1), 10),
        ("PADDING",    (0, 0), (-1, -1), 8),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Recent events table (last 30) ─────────────────────────────────────
    if events:
        story.append(Paragraph("Recent Detection Events (last 30)", section_style))
        recent = events[-30:]
        ev_data = [["Timestamp", "EAR", "MAR", "Drowsy", "Yawn", "Alarm"]]
        for e in recent:
            ev_data.append([
                e["timestamp"][:19],
                f"{e['ear']:.3f}",
                f"{e['mar']:.3f}",
                "YES" if e["drowsy"]          else "no",
                "YES" if e["yawning"]         else "no",
                "YES" if e["alarm_triggered"] else "no",
            ])

        ev_tbl = Table(ev_data,
                       colWidths=[5.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm])
        ev_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#dee2e6")),
            ("PADDING",    (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8f9fa")]),
        ])
        # Highlight drowsy rows in red
        for i, e in enumerate(recent, start=1):
            if e.get("drowsy"):
                ev_style.add("BACKGROUND", (0, i), (-1, i),
                             colors.HexColor("#fdecea"))
                ev_style.add("TEXTCOLOR",  (3, i), (3, i), colors.red)
        ev_tbl.setStyle(ev_style)
        story.append(ev_tbl)

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        "AI Driver Drowsiness Detection System — Confidential Session Report",
        ParagraphStyle("footer", parent=styles["Normal"],
                       fontSize=7, textColor=colors.grey),
    ))

    doc.build(story)
    logger.info("PDF report saved → %s", output_path)
    return output_path
