"""
dashboard.py
============
Real-time Streamlit dashboard for the Driver Drowsiness Detection System.

Run separately from main.py:
    streamlit run dashboard.py

The dashboard reads directly from the SQLite database that main.py writes to,
so both can run simultaneously for live monitoring.
"""

import os
import time
import sqlite3
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import config (works whether run from project root or elsewhere) ──────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "logs", "drowsiness.db"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e3f, #2a2a5a);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #3a3a6a;
        text-align: center;
    }
    .alert-banner {
        background: linear-gradient(135deg, #8b0000, #cc0000);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        margin: 10px 0;
    }
    .status-ok {
        color: #00ff88;
        font-weight: bold;
    }
    .status-warn {
        color: #ff6b35;
        font-weight: bold;
    }
    h1, h2, h3 { color: #89b4fa !important; }
    .stMetric label { color: #cdd6f4 !important; }
</style>
""", unsafe_allow_html=True)


# ── Database helpers ──────────────────────────────────────────────────────────

@st.cache_resource
def get_connection():
    """Return a cached SQLite connection."""
    if not os.path.isfile(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_events(minutes: int = 10) -> pd.DataFrame:
    """Load detection events from the last *minutes* minutes."""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        df = pd.read_sql_query(
            """SELECT * FROM detection_events
               WHERE timestamp >= ?
               ORDER BY timestamp ASC""",
            conn, params=(cutoff,)
        )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


def load_sessions() -> pd.DataFrame:
    """Load all session summaries."""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(
            "SELECT * FROM sessions ORDER BY started_at DESC",
            conn
        )
    except Exception:
        return pd.DataFrame()


def load_latest(n: int = 1) -> dict:
    """Return the single most recent detection event as a dict."""
    conn = get_connection()
    if conn is None:
        return {}
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM detection_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else {}
    except Exception:
        return {}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=60)
    st.title("Drowsiness Monitor")
    st.markdown("---")

    refresh_rate = st.slider("Auto-refresh (seconds)", 1, 10, 3)
    time_window  = st.selectbox("Time window", [2, 5, 10, 30, 60], index=1,
                                 format_func=lambda x: f"Last {x} min")
    show_raw     = st.checkbox("Show raw data table", value=False)

    st.markdown("---")
    st.markdown("**Thresholds**")
    st.markdown(f"EAR alert : `< {config.EAR_THRESHOLD}`")
    st.markdown(f"MAR alert : `> {config.MAR_THRESHOLD}`")
    st.markdown(f"Consec frames : `{config.EAR_CONSEC_FRAMES}`")

    st.markdown("---")
    if st.button("🔄  Refresh Now"):
        st.cache_data.clear()

    st.markdown("---")
    st.caption("Dashboard auto-refreshes. Run `python main.py` to start detection.")


# ── Main content ──────────────────────────────────────────────────────────────

st.title("🚗  AI Driver Drowsiness Detection Dashboard")
st.caption(f"Live data from: `{DB_PATH}`")

# Check DB exists
if not os.path.isfile(DB_PATH):
    st.warning("⚠️  Database not found. Start `python main.py` first to begin logging.")
    st.stop()

# ── Latest status row ─────────────────────────────────────────────────────────
latest = load_latest()

if latest:
    drowsy   = bool(latest.get("drowsy"))
    yawning  = bool(latest.get("yawning"))
    alarm_on = bool(latest.get("alarm_triggered"))

    if alarm_on:
        st.markdown(
            '<div class="alert-banner">⚠️  DROWSINESS ALARM ACTIVE — DRIVER AT RISK!</div>',
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("EAR",  f"{latest.get('ear', 0):.3f}",
              delta="CLOSED" if latest.get("eyes_closed") else "OPEN",
              delta_color="inverse")
    c2.metric("MAR",  f"{latest.get('mar', 0):.3f}",
              delta="YAWNING" if yawning else "normal",
              delta_color="inverse" if yawning else "off")
    c3.metric("FPS",  f"{latest.get('fps', 0):.1f}")
    c4.metric("Status", "😴 DROWSY" if drowsy else "✅ Alert",
              delta_color="inverse")
    c5.metric("Alarm", "🔔 ON" if alarm_on else "🔕 OFF",
              delta_color="inverse" if alarm_on else "off")
else:
    st.info("ℹ️  No detection data yet. Make sure `python main.py` is running.")

st.markdown("---")

# ── Load time-windowed events ─────────────────────────────────────────────────
df = load_events(minutes=time_window)

if df.empty:
    st.warning(f"No events in the last {time_window} minutes.")
else:
    # ── EAR / MAR time-series chart ──────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("👁️  Eye Aspect Ratio (EAR)")
        fig_ear = go.Figure()
        fig_ear.add_trace(go.Scatter(
            x=df["timestamp"], y=df["ear"],
            mode="lines", name="EAR",
            line=dict(color="#89b4fa", width=2),
            fill="tozeroy", fillcolor="rgba(137,180,250,0.1)",
        ))
        fig_ear.add_hline(
            y=config.EAR_THRESHOLD, line_dash="dash",
            line_color="#f38ba8", annotation_text="Drowsy threshold",
        )
        fig_ear.update_layout(
            paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
            font_color="#cdd6f4", height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 0.5], gridcolor="#313244"),
            xaxis=dict(gridcolor="#313244"),
        )
        st.plotly_chart(fig_ear, use_container_width=True)

    with col_right:
        st.subheader("👄  Mouth Aspect Ratio (MAR)")
        fig_mar = go.Figure()
        fig_mar.add_trace(go.Scatter(
            x=df["timestamp"], y=df["mar"],
            mode="lines", name="MAR",
            line=dict(color="#a6e3a1", width=2),
            fill="tozeroy", fillcolor="rgba(166,227,161,0.1)",
        ))
        fig_mar.add_hline(
            y=config.MAR_THRESHOLD, line_dash="dash",
            line_color="#fab387", annotation_text="Yawn threshold",
        )
        fig_mar.update_layout(
            paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
            font_color="#cdd6f4", height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 1.2], gridcolor="#313244"),
            xaxis=dict(gridcolor="#313244"),
        )
        st.plotly_chart(fig_mar, use_container_width=True)

    # ── Event counts bar chart ───────────────────────────────────────────────
    st.subheader("📊  Detection Event Counts")
    col_a, col_b, col_c = st.columns(3)

    drowsy_count = int(df["drowsy"].sum())
    yawn_count   = int(df["yawning"].sum())
    alarm_count  = int(df["alarm_triggered"].sum())

    col_a.metric("Drowsy Frames",  drowsy_count)
    col_b.metric("Yawn Frames",    yawn_count)
    col_c.metric("Alarm Frames",   alarm_count)

    # Timeline heatmap of alerts
    df["minute"] = df["timestamp"].dt.floor("1min")
    alert_df = df.groupby("minute").agg(
        drowsy=("drowsy", "sum"),
        yawning=("yawning", "sum"),
        alarm=("alarm_triggered", "sum"),
    ).reset_index()

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=alert_df["minute"], y=alert_df["drowsy"],
                              name="Drowsy", marker_color="#f38ba8"))
    fig_bar.add_trace(go.Bar(x=alert_df["minute"], y=alert_df["yawning"],
                              name="Yawning", marker_color="#f9e2af"))
    fig_bar.add_trace(go.Bar(x=alert_df["minute"], y=alert_df["alarm"],
                              name="Alarm", marker_color="#cba6f7"))
    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font_color="#cdd6f4", height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="#1e1e2e"),
        xaxis=dict(gridcolor="#313244"),
        yaxis=dict(gridcolor="#313244"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── FPS chart ─────────────────────────────────────────────────────────
    st.subheader("⚡  FPS Performance")
    fig_fps = px.line(df, x="timestamp", y="fps",
                      color_discrete_sequence=["#89dceb"])
    fig_fps.update_layout(
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font_color="#cdd6f4", height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#313244"),
        yaxis=dict(gridcolor="#313244"),
    )
    st.plotly_chart(fig_fps, use_container_width=True)

    # ── Raw data ──────────────────────────────────────────────────────────
    if show_raw:
        st.subheader("📋  Raw Event Log")
        st.dataframe(df.tail(100), use_container_width=True)

# ── Session history ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📁  Session History")
sessions_df = load_sessions()

if not sessions_df.empty:
    display_cols = ["started_at", "ended_at", "total_frames",
                    "drowsy_events", "yawn_events", "alarm_events",
                    "avg_ear", "avg_mar"]
    st.dataframe(
        sessions_df[display_cols].head(10),
        use_container_width=True,
    )

    # Download button for session CSV
    csv = sessions_df.to_csv(index=False)
    st.download_button(
        label="⬇️  Download Session History (CSV)",
        data=csv,
        file_name=f"session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
else:
    st.info("No session history found.")

# ── Head Pose & Night Vision Section ─────────────────────────────────────────
st.markdown("---")
st.subheader("🧭  Head Pose & Night Vision Stats")

if not df.empty:
    col_h1, col_h2, col_h3 = st.columns(3)
    # Latest head state from most recent event (stored as text in stats dict)
    latest_head = load_latest()
    col_h1.metric("Night Vision Active",
                  "ON" if latest_head.get("night_active") else "OFF")

    # Show head alert rate
    if "head_alert" in df.columns:
        alert_rate = df["head_alert"].mean() * 100 if "head_alert" in df.columns else 0
        col_h2.metric("Head Alert Rate", f"{alert_rate:.1f}%")

    nv_stats = {
        "Night Vision": "AUTO mode — enhances when brightness < threshold",
        "CLAHE Mode": "Very dark environments — adaptive histogram equalization",
        "Gamma Mode": "Moderately dim — power-law brightness correction",
    }
    for k, v in nv_stats.items():
        st.caption(f"**{k}**: {v}")
else:
    st.info("Start detection to see head pose and night vision data.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"⏱ Last updated: {datetime.now().strftime('%H:%M:%S')} — "
           f"refreshing every {refresh_rate}s")
time.sleep(refresh_rate)
st.rerun()
