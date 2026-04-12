from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pipeline.analyzer import ConversationAnalyzer
from pipeline.insights import analyze_insights
from pipeline.visualizer import build_visualization_data
from utils.parser import parse_conversation


warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="FR3AK Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

plt.style.use("dark_background")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #0b0b0f;
            --bg-secondary: #161324;
            --purple-main: #6c3cff;
            --purple-soft: #9a7dff;
            --text-main: #f2f3f8;
            --text-muted: #bec2d9;
            --glass: rgba(255, 255, 255, 0.06);
            --glass-border: rgba(180, 152, 255, 0.28);
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(108, 60, 255, 0.18), transparent 35%),
                radial-gradient(circle at 90% 5%, rgba(108, 60, 255, 0.12), transparent 40%),
                linear-gradient(135deg, var(--bg-primary), #120f1d 45%, #1a1230 100%);
            color: var(--text-main);
            font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 1.5rem;
            max-width: 1300px;
        }

        .global-hero {
            text-align: center;
            margin-bottom: 0.9rem;
        }

        .global-title {
            margin: 0;
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: 0.02em;
            background: linear-gradient(90deg, #d8c4ff, #b48cff, #82a4ff);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 0 0 20px rgba(140, 99, 255, 0.25);
        }

        .global-subtitle {
            margin-top: 0.35rem;
            color: var(--text-muted);
            font-size: 0.95rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .fr3ak-title {
            font-size: 2.3rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.25rem;
            color: var(--text-main);
        }

        .fr3ak-subtitle {
            font-size: 1rem;
            color: var(--text-muted);
            margin-bottom: 1.3rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .glass-card {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
            border: 1px solid var(--glass-border);
            border-radius: 18px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 10px 32px rgba(0, 0, 0, 0.35), 0 0 18px rgba(108, 60, 255, 0.14);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            margin-bottom: 0.75rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
            border: 1px solid var(--glass-border);
            border-radius: 18px;
            box-shadow: 0 10px 32px rgba(0, 0, 0, 0.35), 0 0 18px rgba(108, 60, 255, 0.14);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0.95rem 1.1rem;
        }

        .metric-tag {
            display: inline-block;
            margin-right: 0.4rem;
            margin-bottom: 0.35rem;
            padding: 0.26rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            color: var(--text-main);
            border: 1px solid rgba(171, 145, 255, 0.55);
            background: rgba(108, 60, 255, 0.22);
        }

        .tone-positive { border-color: rgba(102, 255, 182, 0.55); background: rgba(72, 190, 146, 0.20); }
        .tone-mixed { border-color: rgba(255, 214, 120, 0.55); background: rgba(255, 189, 79, 0.20); }
        .tone-negative { border-color: rgba(255, 114, 114, 0.6); background: rgba(255, 89, 89, 0.20); }
        .tone-dynamic { border-color: rgba(126, 196, 255, 0.6); background: rgba(69, 154, 255, 0.20); }

        .stButton > button {
            background: linear-gradient(135deg, rgba(108, 60, 255, 0.95), rgba(142, 95, 255, 0.95));
            color: #f7f8ff;
            border-radius: 12px;
            border: 1px solid rgba(181, 155, 255, 0.58);
            font-weight: 600;
            padding: 0.48rem 0.95rem;
            transition: all 0.2s ease;
            box-shadow: 0 6px 16px rgba(80, 43, 190, 0.35);
        }

        .stButton > button:hover {
            transform: translateY(-1px) scale(1.01);
            box-shadow: 0 8px 22px rgba(115, 67, 255, 0.45), 0 0 0 1px rgba(208, 194, 255, 0.28) inset;
        }

        .section-title {
            margin: 0.2rem 0 0.6rem;
            font-size: 1.07rem;
            font-weight: 650;
            color: var(--text-main);
        }

        .hero-panel {
            display: grid;
            gap: 0.65rem;
            padding: 0.2rem 0 0.15rem;
        }

        .hero-copy {
            color: var(--text-muted);
            font-size: 0.95rem;
            line-height: 1.5;
            max-width: 62rem;
        }

        .summary-text {
            color: var(--text-main);
            line-height: 1.6;
        }

        .wheel-panel {
            min-height: 420px;
        }

        .small-note {
            color: var(--text-muted);
            font-size: 0.86rem;
        }

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px dashed rgba(177, 150, 255, 0.4);
            border-radius: 14px;
            padding: 0.5rem 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_global_header() -> None:
    st.markdown(
        """
        <div class="global-hero">
            <h1 class="global-title">Functional Reasoning & Emotional Augmentation Kernel</h1>
            <div class="global-subtitle">Emotion • Behavior • Insight</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_session() -> None:
    defaults: Dict[str, Any] = {
        "page": "home",
        "raw_records": [],
        "analysis_output": None,
        "insights_output": None,
        "viz_data": None,
        "selected_user": None,
        "processed_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_analyzer() -> ConversationAnalyzer:
    return ConversationAnalyzer()


def _normalize_json_records(payload: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            payload = payload["records"]
        elif isinstance(payload.get("messages"), list):
            payload = payload["messages"]
        else:
            payload = []

    if not isinstance(payload, list):
        return records

    for item in payload:
        if not isinstance(item, Mapping):
            continue
        speaker = str(item.get("speaker") or item.get("speaker_id") or "").strip()
        cleaned_message = str(item.get("cleaned_message") or item.get("message") or item.get("text") or "").strip()
        timestamp = item.get("timestamp", None)
        if not speaker or not cleaned_message:
            continue
        records.append(
            {
                "speaker": speaker,
                "cleaned_message": cleaned_message,
                "timestamp": timestamp,
            }
        )

    return records


def load_data(uploaded_file: Any) -> Tuple[List[Dict[str, Any]], str]:
    if uploaded_file is None:
        return [], ""

    ext = Path(uploaded_file.name).suffix.lower()

    if ext == ".txt":
        raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        parsed = parse_conversation(raw_text)
        records: List[Dict[str, Any]] = []
        for record in parsed.records:
            speaker = str(getattr(record, "speaker", "") or getattr(record, "speaker_id", "")).strip()
            cleaned_message = str(getattr(record, "cleaned_message", "") or "").strip()
            timestamp_obj = getattr(record, "timestamp", None)
            timestamp = getattr(timestamp_obj, "raw", None) if timestamp_obj else None
            if not speaker or not cleaned_message:
                continue
            records.append(
                {
                    "speaker": speaker,
                    "cleaned_message": cleaned_message,
                    "timestamp": timestamp,
                }
            )
        return records, "txt"

    if ext == ".json":
        payload = json.loads(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
        return _normalize_json_records(payload), "json"

    return [], ext


def run_pipeline(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    analyzer = get_analyzer()
    analysis_output = analyzer.analyze(records)
    insights_output = analyze_insights(analysis_output)
    viz_data = build_visualization_data(analysis_output, insights_output)
    return analysis_output, insights_output, viz_data


def _tone_class(tone: str) -> str:
    normalized = str(tone or "").strip().lower()
    if normalized in {"positive", "mixed", "negative", "dynamic"}:
        return f"tone-{normalized}"
    return "tone-mixed"


def _user_messages(analysis_output: Mapping[str, Any], speaker: str) -> List[Mapping[str, Any]]:
    raw_messages = analysis_output.get("messages", []) if isinstance(analysis_output, Mapping) else []
    if not isinstance(raw_messages, list):
        return []

    speaker_messages = [m for m in raw_messages if isinstance(m, Mapping) and str(m.get("speaker", "")) == speaker]
    speaker_messages.sort(key=lambda m: int(m.get("index", 0)) if str(m.get("index", "")).isdigit() else 0)
    return speaker_messages


def _get_user_colors(users: List[str]) -> Dict[str, str]:
    palette = [
        "#8a6cff",  # violet
        "#5b8dff",  # blue-violet
        "#d47bff",  # pink-purple
        "#7bb4ff",  # cool blue
        "#b689ff",  # lavender
        "#ff8dd8",  # magenta soft
    ]
    mapping = dict(st.session_state.get("user_colors", {}))
    for user in sorted(users):
        if user not in mapping:
            mapping[user] = palette[len(mapping) % len(palette)]
    st.session_state.user_colors = mapping
    return mapping


def _plot_emotion_trend_for_user(messages: List[Mapping[str, Any]], speaker: str, user_color: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    fig.patch.set_facecolor("#0f0d18")
    ax.set_facecolor("#171327")
    xs = [int(msg.get("index", idx)) for idx, msg in enumerate(messages)]
    ys = [float(msg.get("emotion_intensity", 0.0)) for msg in messages]

    ax.plot(xs, ys, marker="o", linewidth=2.5, markersize=6.5, color=user_color)
    ax.set_title(f"Emotion Trend - {speaker}")
    ax.set_xlabel("Message Index")
    ax.set_ylabel("Emotion Intensity (Normalized)")
    ax.set_ylim(0.0, 1.1)
    ax.grid(True, linestyle="--", alpha=0.22, color="#bca7ff")
    for spine in ax.spines.values():
        spine.set_color("#6a5a9f")
    fig.tight_layout()
    return fig


def _plot_binary_timeline(
    messages: List[Mapping[str, Any]],
    speaker: str,
    field: str,
    title: str,
    user_color: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    fig.patch.set_facecolor("#0f0d18")
    ax.set_facecolor("#171327")
    xs = [int(msg.get("index", idx)) for idx, msg in enumerate(messages)]

    if field == "is_sarcastic":
        ys = [1.0 if bool(msg.get("is_sarcastic", False)) else 0.0 for msg in messages]
        ylabel = "Sarcasm (0/1)"
    else:
        ys = [1.0 if str(msg.get("label", "genuine")).lower() == "manipulative" else 0.0 for msg in messages]
        ylabel = "Manipulation (0/1)"

    ax.scatter(xs, ys, color=user_color, s=88, alpha=0.92, edgecolors="#2f2542", linewidths=0.55)
    ax.set_title(f"{title} - {speaker}")
    ax.set_xlabel("Message Index")
    ax.set_ylabel(ylabel)
    ax.set_yticks([0, 1])
    ax.grid(True, linestyle="--", alpha=0.22, color="#bca7ff")
    for spine in ax.spines.values():
        spine.set_color("#6a5a9f")
    fig.tight_layout()
    return fig


def _plot_plutchik_wheel(emotion_vector: List[float], speaker: str, user_color: str) -> plt.Figure:
    values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in emotion_vector]
    if len(values) < 8:
        values.extend([0.0] * (8 - len(values)))
    elif len(values) > 8:
        values = values[:8]

    values_array = np.asarray(values, dtype=float)
    values_array = np.clip(values_array, 0.0, None)
    peak = float(values_array.max()) if values_array.size else 0.0
    if peak > 0:
        display_values = 0.18 + 0.82 * (values_array / peak)
    else:
        display_values = np.full_like(values_array, 0.18)

    labels = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    theta_closed = np.r_[theta, theta[0]]
    values_closed = np.r_[display_values, display_values[0]]

    dense_theta = np.linspace(0, 2 * np.pi, 360)
    dense_values = np.interp(dense_theta, theta_closed, values_closed)

    fig = plt.figure(figsize=(7.0, 5.2), facecolor="#0f0d18")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#121025")

    ax.set_rlim(0, 1.12)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_rlabel_position(180)

    ax.bar(
        theta,
        display_values,
        width=(2 * np.pi / len(labels)) * 0.82,
        bottom=0.08,
        color=user_color,
        alpha=0.18,
        edgecolor="#f5edff",
        linewidth=0.9,
        align="center",
        zorder=2,
    )

    for width, alpha in ((10, 0.07), (6.2, 0.12), (3.5, 0.18)):
        ax.plot(dense_theta, dense_values, color=user_color, linewidth=width, alpha=alpha)

    ax.plot(dense_theta, dense_values, color="#f5edff", linewidth=2.3)

    for step, alpha in ((1.00, 0.36), (0.75, 0.25), (0.50, 0.15), (0.25, 0.09)):
        ax.fill(dense_theta, dense_values * step, color=user_color, alpha=alpha)

    center_theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(center_theta, np.full_like(center_theta, 0.15), color="#ffffff", alpha=0.06)

    ax.scatter(theta, display_values, s=38, color="#ffffff", alpha=0.82, zorder=4)

    ax.set_xticks(theta)
    ax.set_xticklabels(labels, color="#f1ecff", fontsize=10, fontweight="semibold")
    ax.tick_params(pad=10)
    ax.set_yticklabels([])
    ax.grid(alpha=0.34, color="#9f8ae0", linewidth=0.8)
    ax.spines["polar"].set_color(user_color)
    ax.set_title(f"Plutchik Emotion Wheel - {speaker}", color="#f5f2ff", pad=16)

    fig.tight_layout()
    return fig


def render_home() -> None:
    st.markdown('<div class="section-title">Start an analysis run</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Upload a raw conversation export in <strong>.txt</strong> or <strong>.json</strong>, then run the FR3AK pipeline to generate user profiles, emotional trends, and behavior signals.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        uploaded = st.file_uploader("Upload conversation file (.txt or .json)", type=["txt", "json"])

    if uploaded is not None:
        signature = (uploaded.name, len(uploaded.getvalue()))
        already_processed = st.session_state.get("processed_signature") == signature

        run_col, nav_col = st.columns([1.4, 1])
        with run_col:
            if st.button("Run FR3AK Pipeline", type="primary"):
                with st.spinner("Processing conversation through parser, analyzer, and insights..."):
                    records, source_type = load_data(uploaded)
                    if not records:
                        st.warning("No valid records found in uploaded file.")
                    else:
                        analysis_output, insights_output, viz_data = run_pipeline(records)
                        st.session_state.raw_records = records
                        st.session_state.analysis_output = analysis_output
                        st.session_state.insights_output = insights_output
                        st.session_state.viz_data = viz_data
                        st.session_state.processed_signature = signature
                        st.session_state.selected_user = None
                        st.session_state.page = "user_select"
                        st.success(f"Pipeline completed successfully from {source_type.upper()} input.")
                        st.rerun()

        with nav_col:
            if already_processed:
                st.caption("File already processed. You can go to user selection directly.")
                if st.button("Go To User Selection"):
                    st.session_state.page = "user_select"
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_user_select() -> None:
    insights = st.session_state.insights_output or {}

    st.markdown('<div class="section-title">Select a user for deep analysis</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    back_col, spacer = st.columns([1, 5])
    with back_col:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()

    users = [u for u in insights.keys() if isinstance(u, str)]
    user_colors = _get_user_colors(users)
    if not users:
        st.info("No users found. Upload and process a conversation from Home first.")
        return

    cols = st.columns(2)
    for idx, user in enumerate(users):
        profile = insights.get(user, {}) if isinstance(insights.get(user, {}), Mapping) else {}
        tone = str(profile.get("emotional_tone", "mixed"))
        sarcasm = str(profile.get("sarcasm_level", "low"))
        manipulation = str(profile.get("manipulation_level", "low"))
        accent = user_colors.get(user, "#8a6cff")

        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"### <span style='color:{accent}'>{user}</span>", unsafe_allow_html=True)
                st.markdown(
                    f'<span class="metric-tag {_tone_class(tone)}" style="border-color:{accent}77;background:{accent}2b;">tone: {tone}</span>'
                    f'<span class="metric-tag" style="border-color:{accent}77;background:{accent}22;">sarcasm: {sarcasm}</span>'
                    f'<span class="metric-tag" style="border-color:{accent}77;background:{accent}22;">manipulation: {manipulation}</span>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Open {user}", key=f"open-user-{user}"):
                    st.session_state.selected_user = user
                    st.session_state.page = "user_analysis"
                    st.rerun()


def render_user_analysis() -> None:
    analysis = st.session_state.analysis_output or {}
    insights = st.session_state.insights_output or {}
    selected_user = st.session_state.selected_user

    st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
    spacer_left, nav_home, nav_users, spacer_right = st.columns([2, 1, 1, 2])
    with nav_home:
        if st.button("Home"):
            st.session_state.page = "home"
            st.rerun()
    with nav_users:
        if st.button("Users"):
            st.session_state.page = "user_select"
            st.rerun()

    if not selected_user or selected_user not in insights:
        st.info("No user selected. Return to user selection.")
        return

    profile = insights.get(selected_user, {}) if isinstance(insights.get(selected_user, {}), Mapping) else {}
    user_colors = _get_user_colors([u for u in insights.keys() if isinstance(u, str)])
    user_color = user_colors.get(selected_user, "#8a6cff")
    messages = _user_messages(analysis, selected_user)
    user_stats = (analysis.get("users", {}) or {}).get(selected_user, {}) if isinstance(analysis, Mapping) else {}
    emotion_avg = user_stats.get("emotion_avg", []) if isinstance(user_stats, Mapping) else []

    st.markdown('<div class="section-title">User Profile</div>', unsafe_allow_html=True)
    st.markdown(f"## <span style='color:{user_color}'>{selected_user}</span>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            f'<div class="summary-text" style="max-width: 800px;"><strong>Summary</strong><br>{profile.get("summary", "No summary available.")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="metric-tag {_tone_class(str(profile.get("emotional_tone", "mixed")))}" style="border-color:{user_color}77;background:{user_color}2b;">tone: {profile.get("emotional_tone", "mixed")}</span>'
            f'<span class="metric-tag" style="border-color:{user_color}77;background:{user_color}22;">sarcasm: {profile.get("sarcasm_level", "low")}</span>'
            f'<span class="metric-tag" style="border-color:{user_color}77;background:{user_color}22;">manipulation: {profile.get("manipulation_level", "low")}</span>'
            f'<span class="metric-tag" style="border-color:{user_color}77;background:{user_color}22;">neutral: {profile.get("neutral_level", "low")}</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Visual Analysis</div>', unsafe_allow_html=True)
    grid_top = st.columns(2)
    with grid_top[0]:
        with st.container(border=True):
            st.pyplot(_plot_emotion_trend_for_user(messages, selected_user, user_color), clear_figure=True)
    with grid_top[1]:
        with st.container(border=True):
            st.pyplot(_plot_binary_timeline(messages, selected_user, "is_sarcastic", "Sarcasm Timeline", user_color), clear_figure=True)

    grid_bottom = st.columns(2)
    with grid_bottom[0]:
        with st.container(border=True):
            st.pyplot(_plot_binary_timeline(messages, selected_user, "label", "Manipulation Timeline", user_color), clear_figure=True)
    with grid_bottom[1]:
        st.markdown('<div class="section-title">Emotional Signature</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div style='margin-top: 10px'></div>", unsafe_allow_html=True)
            fig = _plot_plutchik_wheel(
                emotion_avg if isinstance(emotion_avg, list) else [],
                selected_user,
                user_color,
            )
            st.pyplot(fig, clear_figure=True)


inject_styles()
init_session()
render_global_header()

page = st.session_state.page
if page == "home":
    render_home()
elif page == "user_select":
    render_user_select()
elif page == "user_analysis":
    render_user_analysis()
else:
    st.session_state.page = "home"
    render_home()
