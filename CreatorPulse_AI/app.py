import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path
from datetime import datetime

# Import modules from src
from src.engagement_metrics import detect_anomalies, compute_content_dna
from src.virality_prediction import (
    load_virality_model,
    compute_feature_contributions,
    ALL_FEATURES
)
from src.recommendations import generate_recommendations
from src.ab_testing import compute_format_comparison

# 1. PROJECT PATHS & CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"
EXPERIMENTS_FILE = PROCESSED_DIR / "experiments.json"

CONTENT_FILE = PROCESSED_DIR / "content_engagement_metrics.csv"
COMMENTS_FILE = PROCESSED_DIR / "analyzed_comments.csv"
AB_FILE = PROCESSED_DIR / "ab_testing_results.csv"
TREND_FILE = REPORT_DIR / "trend_forecast.csv"

# Streamlit Page Setup
st.set_page_config(
    page_title="CreatorPulse AI | Commercial Social Media Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ADVANCED COMMERCIAL SAAS STYLING (Midnight Navy, Aurora Purple, Electric Blue & Cyan)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Deep Midnight Canvas */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #060813;
        background-image: 
            radial-gradient(circle at 12% 12%, rgba(124, 58, 237, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 88% 18%, rgba(14, 165, 233, 0.10) 0%, transparent 40%),
            radial-gradient(circle at 50% 65%, rgba(99, 102, 241, 0.05) 0%, transparent 60%);
        background-attachment: fixed;
        color: #f8fafc;
    }

    /* Branded SaaS Sidebar */
    [data-testid="stSidebar"] {
        background: #080b16 !important;
        border-right: 1px solid rgba(139, 92, 246, 0.18) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(139, 92, 246, 0.15) !important;
        margin: 16px 0 !important;
    }

    /* Brand Box in Sidebar */
    .sidebar-brand-box {
        padding: 6px 4px 16px 4px;
        border-bottom: 1px solid rgba(139, 92, 246, 0.14);
        margin-bottom: 16px;
    }
    .brand-title-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }
    .brand-icon-box {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #7c3aed 0%, #0284c7 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        box-shadow: 0 0 16px rgba(124, 58, 237, 0.45);
    }
    .brand-title-text {
        font-size: 20px;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #c084fc 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .brand-tagline {
        font-size: 11px;
        color: #94a3b8;
        font-weight: 500;
        letter-spacing: 0.02em;
        margin-top: 4px;
        line-height: 1.3;
    }
    .status-indicator-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        padding: 3px 10px;
        font-size: 10px;
        font-weight: 600;
        color: #34d399;
        margin-top: 10px;
        letter-spacing: 0.04em;
    }
    .pulse-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Sidebar Navigation Links Upgrade */
    div[data-testid="stRadio"] > div {
        gap: 4px;
    }
    div[data-testid="stRadio"] label {
        background: rgba(15, 23, 42, 0.4);
        border-radius: 8px;
        padding: 8px 12px;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
        border: 1px solid rgba(255, 255, 255, 0.04);
        color: #cbd5e1;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(139, 92, 246, 0.12);
        border-color: rgba(139, 92, 246, 0.35);
        color: #f8fafc;
        transform: translateX(3px);
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.15);
    }

    /* Sidebar Glass Filter Container */
    .filter-panel {
        background: rgba(14, 20, 39, 0.65);
        border: 1px solid rgba(139, 92, 246, 0.18);
        border-radius: 12px;
        padding: 14px;
        margin-top: 10px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    .filter-panel-title {
        font-size: 11px;
        font-weight: 700;
        color: #c084fc;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Premium Welcome Banner Header */
    .welcome-header-card {
        background: linear-gradient(135deg, rgba(17, 24, 48, 0.85) 0%, rgba(10, 14, 29, 0.95) 100%);
        border: 1px solid rgba(139, 92, 246, 0.24);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5), 0 0 20px rgba(124, 58, 237, 0.1);
    }
    .welcome-header-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #8b5cf6 0%, #38bdf8 50%, #06b6d4 100%);
    }
    .welcome-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 8px;
    }
    .welcome-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.35);
        color: #c084fc;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .welcome-chips {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .header-chip {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 11px;
        color: #94a3b8;
        font-weight: 500;
    }
    .header-chip.accent {
        border-color: rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.08);
    }
    .welcome-main-title {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.025em;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .welcome-main-title .gradient-text {
        background: linear-gradient(135deg, #c084fc 0%, #60a5fa 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .welcome-sub {
        font-size: 13px;
        color: #94a3b8;
        margin: 0;
        max-width: 900px;
        line-height: 1.5;
    }

    /* Luxury KPI Cards with Mini Sparklines */
    .kpi-card {
        background: linear-gradient(145deg, rgba(14, 20, 39, 0.85) 0%, rgba(10, 14, 28, 0.95) 100%);
        border: 1px solid rgba(139, 92, 246, 0.18);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 128px;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-color, #8b5cf6), #38bdf8);
        opacity: 0.6;
        transition: opacity 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(168, 85, 247, 0.45);
        box-shadow: 0 12px 32px -4px rgba(0, 0, 0, 0.65), 0 0 20px rgba(139, 92, 246, 0.18);
    }
    .kpi-card:hover::before {
        opacity: 1;
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-icon-pill {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }
    .kpi-body {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 12px;
        margin: 4px 0;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 40%, var(--accent-color, #c084fc) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .kpi-sparkline {
        flex-shrink: 0;
        display: flex;
        align-items: center;
    }
    .kpi-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Refined Glass Card Containers */
    .glass-card {
        background: linear-gradient(145deg, rgba(14, 20, 39, 0.8) 0%, rgba(9, 13, 26, 0.9) 100%);
        border: 1px solid rgba(139, 92, 246, 0.18);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
        transition: all 0.25s ease;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #8b5cf6 0%, #38bdf8 50%, #06b6d4 100%);
        opacity: 0.5;
    }
    .glass-card:hover {
        border-color: rgba(168, 85, 247, 0.35);
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.55), 0 0 20px rgba(139, 92, 246, 0.12);
    }

    /* Section Subheaders */
    .section-header-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 6px;
        margin-bottom: 14px;
    }
    .section-pill-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #38bdf8;
        box-shadow: 0 0 10px #38bdf8;
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
        margin: 0;
    }

    /* Status Badges */
    .badge-synthetic {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }
    .badge-caveat {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }
    .badge-verified {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.14);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.38);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.02em;
    }

    /* Streamlit Buttons & Form Submit Buttons */
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #7c3aed 0%, #2563eb 50%, #0284c7 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.22s ease !important;
        letter-spacing: 0.02em !important;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(124, 58, 237, 0.6), 0 0 16px rgba(56, 189, 248, 0.35) !important;
    }

    /* Native Metric Overrides */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(14, 20, 39, 0.85) 0%, rgba(10, 14, 28, 0.95) 100%);
        border: 1px solid rgba(139, 92, 246, 0.18);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #ffffff 0%, #c084fc 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    /* Streamlit Progress Bar (Electric Purple to Blue Gradient) */
    div[data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #8b5cf6 0%, #38bdf8 100%) !important;
        border-radius: 9999px !important;
    }

    /* Dataframe Container Polish */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(139, 92, 246, 0.18) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background: rgba(10, 14, 28, 0.6) !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        color: #94a3b8 !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #c084fc !important;
        background: rgba(139, 92, 246, 0.12) !important;
        border-bottom-color: #8b5cf6 !important;
    }

    /* Input Controls Polish */
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border-color: rgba(139, 92, 246, 0.25) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] > div {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border-color: rgba(139, 92, 246, 0.25) !important;
        border-radius: 8px !important;
    }

    /* Custom Dividers */
    hr {
        border-color: rgba(139, 92, 246, 0.14) !important;
        margin: 22px 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# 3. HELPER VISUALIZATION & KPI COMPONENTS
def get_sparkline_svg(color="#8b5cf6", trend="up"):
    """Generate lightweight inline SVG mini-sparklines for KPI cards."""
    if trend == "up":
        points = "0,28 15,24 30,26 45,18 60,20 75,12 90,15 105,8 120,4"
        area = "0,28 15,24 30,26 45,18 60,20 75,12 90,15 105,8 120,4 120,32 0,32"
    elif trend == "stable":
        points = "0,18 15,20 30,14 45,17 60,15 75,19 90,13 105,16 120,15"
        area = "0,18 15,20 30,14 45,17 60,15 75,19 90,13 105,16 120,15 120,32 0,32"
    elif trend == "growth":
        points = "0,30 20,26 40,22 60,16 80,18 100,8 120,3"
        area = "0,30 20,26 40,22 60,16 80,18 100,8 120,3 120,32 0,32"
    else:
        points = "0,16 20,12 40,20 60,10 80,22 100,14 120,18"
        area = "0,16 20,12 40,20 60,10 80,22 100,14 120,18 120,32 0,32"

    uid = color.replace("#", "")
    return f"""<svg width="104" height="30" viewBox="0 0 120 32" fill="none" xmlns="http://www.w3.org/2000/svg" style="overflow: visible;">
        <defs>
            <linearGradient id="grad_{uid}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="{color}" stop-opacity="0.32"/>
                <stop offset="100%" stop-color="{color}" stop-opacity="0.0"/>
            </linearGradient>
        </defs>
        <polygon points="{area}" fill="url(#grad_{uid})" />
        <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="120" cy="{points.split()[-1].split(',')[1]}" r="3" fill="{color}" />
    </svg>"""


def render_kpi_card(label, value, subtext, icon="⚡", color="#8b5cf6", trend="up"):
    """Render a luxury SaaS KPI Card with inline SVG mini sparkline."""
    sparkline = get_sparkline_svg(color=color, trend=trend)
    return f"""
    <div class="kpi-card" style="--accent-color: {color};">
        <div class="kpi-header">
            <span class="kpi-label">{label}</span>
            <span class="kpi-icon-pill" style="color: {color}; background: {color}18; border: 1px solid {color}33;">{icon}</span>
        </div>
        <div class="kpi-body">
            <div class="kpi-value">{value}</div>
            <div class="kpi-sparkline">{sparkline}</div>
        </div>
        <div class="kpi-sub">{subtext}</div>
    </div>
    """


def apply_plotly_theme(fig, height=None, show_legend=True):
    """Apply consistent midnight navy, aurora purple and electric blue styling to Plotly figures."""
    layout_kwargs = dict(
        paper_bgcolor="rgba(10, 14, 30, 0.75)",
        plot_bgcolor="rgba(13, 19, 38, 0.4)",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#e2e8f0", size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="rgba(139, 92, 246, 0.5)",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#ffffff")
        )
    )
    if height:
        layout_kwargs["height"] = height
    if show_legend:
        layout_kwargs["legend"] = dict(
            bgcolor="rgba(11, 15, 31, 0.75)",
            bordercolor="rgba(139, 92, 246, 0.2)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1", size=11)
        )
    else:
        layout_kwargs["showlegend"] = False

    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(139, 92, 246, 0.08)",
        zeroline=False,
        tickfont=dict(color="#94a3b8", size=11)
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(139, 92, 246, 0.08)",
        zeroline=False,
        tickfont=dict(color="#94a3b8", size=11)
    )
    return fig


# 4. DATA CACHING & LOADERS
@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


content_df = load_csv(CONTENT_FILE)
comments_df = load_csv(COMMENTS_FILE)
ab_df = load_csv(AB_FILE)
trend_df = load_csv(TREND_FILE)


def load_experiments():
    if EXPERIMENTS_FILE.exists():
        try:
            with open(EXPERIMENTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_experiments(experiments):
    EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_FILE, "w") as f:
        json.dump(experiments, f, indent=2)


# 5. SIDEBAR BRANDING & NAVIGATION
st.sidebar.markdown("""
<div class="sidebar-brand-box">
    <div class="brand-title-wrap">
        <div class="brand-icon-box">⚡</div>
        <div>
            <div class="brand-title-text">CreatorPulse AI</div>
            <div class="brand-tagline">Data-Driven Social Intelligence</div>
        </div>
    </div>
    <div class="status-indicator-pill">
        <span class="pulse-dot"></span>
        <span>STUDIO ENGINE ACTIVE • DEMO</span>
    </div>
</div>
""", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Overview & Executive KPIs",
        "Engagement Analysis & A/B Testing",
        "Audience Sentiment Intelligence",
        "Virality Prediction & Explainable AI",
        "What-If Content Simulator",
        "Content DNA Analyzer",
        "Engagement Anomaly Detection",
        "Trend Forecasting & Baselines",
        "Strategy & Experiment Planner"
    ]
)

# Sidebar Filter Controls inside Glass Container
st.sidebar.markdown("""
<div class="filter-panel">
    <div class="filter-panel-title">
        <span>🔍</span> Live Dataset Filters
    </div>
</div>
""", unsafe_allow_html=True)

if not content_df.empty:
    all_formats = ["All"] + sorted(content_df["format"].unique().tolist())
    selected_format = st.sidebar.selectbox("Filter by Format", all_formats)

    all_topics = ["All"] + sorted(content_df["topic"].unique().tolist())
    selected_topic = st.sidebar.selectbox("Filter by Topic", all_topics)

    # Filter dataframe
    filtered_content = content_df.copy()
    if selected_format != "All":
        filtered_content = filtered_content[filtered_content["format"] == selected_format]
    if selected_topic != "All":
        filtered_content = filtered_content[filtered_content["topic"] == selected_topic]
else:
    filtered_content = pd.DataFrame()

st.sidebar.caption("Data provenance: Synthetic academic sample (50 posts, 150 comments). Verified local execution.")

# 6. SHARED WELCOME HEADER BANNER
st.markdown("""
<div class="welcome-header-card">
    <div class="welcome-top-row">
        <div class="welcome-badge">
            <span class="pulse-dot"></span>
            <span>Enterprise Analytics Studio</span>
        </div>
        <div class="welcome-chips">
            <span class="header-chip">50 Posts Sample</span>
            <span class="header-chip">150 Comments</span>
            <span class="header-chip accent">Random Forest & NLP</span>
        </div>
    </div>
    <div class="welcome-main-title">Data-Driven <span class="gradient-text">Social Engagement Initiative</span></div>
    <p class="welcome-sub">CreatorPulse AI • Content Performance Intelligence, Explainable Prediction & Controlled Experimentation</p>
</div>
""", unsafe_allow_html=True)

# PAGE 1: OVERVIEW & EXECUTIVE KPIS


if nav_page == "Overview & Executive KPIs":
    st.markdown("""
    <div class="section-header-wrap">
        <div class="section-pill-dot"></div>
        <div class="section-title">Creator Performance Overview & Executive KPIs</div>
    </div>
    """, unsafe_allow_html=True)

    if filtered_content.empty:
        st.warning("No data available under the selected filters.")
    else:
        total_posts = len(filtered_content)
        total_views = filtered_content["views"].sum()
        total_interactions = filtered_content["total_interactions"].sum()
        macro_avg = filtered_content["engagement_rate"].mean()
        weighted_avg = (total_interactions / total_views * 100) if total_views > 0 else 0

        # KPI Cards with Mini Sparklines
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                render_kpi_card(
                    label="Macro Avg Engagement",
                    value=f"{macro_avg:.2f}%",
                    subtext=f"Mean of post rates (N={total_posts})",
                    icon="📈",
                    color="#c084fc",
                    trend="up"
                ),
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                render_kpi_card(
                    label="Weighted Engagement",
                    value=f"{weighted_avg:.2f}%",
                    subtext=f"{total_interactions:,.0f} interactions / {total_views:,.0f} views",
                    icon="⚖️",
                    color="#38bdf8",
                    trend="stable"
                ),
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                render_kpi_card(
                    label="Total Verified Views",
                    value=f"{total_views:,.0f}",
                    subtext=f"Across {total_posts} sample posts",
                    icon="👁️",
                    color="#22d3ee",
                    trend="growth"
                ),
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                render_kpi_card(
                    label="Total Interactions",
                    value=f"{total_interactions:,.0f}",
                    subtext="Likes, shares, saves & comments",
                    icon="💬",
                    color="#818cf8",
                    trend="up"
                ),
                unsafe_allow_html=True
            )

        st.write("")

        # Engagement Timeline Plot
        st.markdown("#### Historical Engagement Timeline")
        filtered_content_sorted = filtered_content.copy()
        filtered_content_sorted["post_date"] = pd.to_datetime(filtered_content_sorted["post_date"])
        filtered_content_sorted = filtered_content_sorted.sort_values("post_date")

        fig_timeline = px.scatter(
            filtered_content_sorted,
            x="post_date",
            y="engagement_rate",
            color="format",
            size="views",
            color_discrete_map={
                "Short": "#a855f7",
                "Image": "#38bdf8",
                "Carousel": "#06b6d4",
                "Reel": "#6366f1"
            },
            hover_data=["post_id", "topic", "hook_type", "total_interactions"],
            labels={"post_date": "Post Date", "engagement_rate": "Engagement Rate (%)", "format": "Content Format"},
            title="Engagement Rate by Date, Format, and Audience Reach"
        )
        apply_plotly_theme(fig_timeline)
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Top Performing Posts vs Low Performing Posts
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Top 5 Performing Posts")
            top5 = filtered_content.sort_values("engagement_rate", ascending=False).head(5)[
                ["post_id", "topic", "format", "hook_type", "views", "engagement_rate", "engagement_rank"]
            ]
            st.dataframe(top5, use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("#### Data Provenance & Methodological Note")
            st.info(
                "**Sample Profile:** Synthetic benchmark dataset containing exactly 50 social media posts spanning Sept 1, 2026 to Dec 8, 2026.\n\n"
                "**Metric Definition:** `Engagement Rate = (Likes + Shares + Saves + Comments) / Views * 100`.\n\n"
                "**Guardrail:** Observed patterns are exploratory sample associations. They do not demonstrate platform-wide causality or guaranteed reach."
            )

# PAGE 2: ENGAGEMENT ANALYSIS & A/B TESTING

elif nav_page == "Engagement Analysis & A/B Testing":
    st.markdown("""
    <div class="section-header-wrap">
        <div class="section-pill-dot"></div>
        <div class="section-title">Content Performance & A/B Format Comparison</div>
    </div>
    """, unsafe_allow_html=True)

    if filtered_content.empty:
        st.warning("No data found for selected filters.")
    else:
        # Scope Description
        scope_parts = []
        if selected_topic != "All":
            scope_parts.append(f"Topic: **{selected_topic}**")
        if selected_format != "All":
            scope_parts.append(f"Format: **{selected_format}**")
        scope_label = ", ".join(scope_parts) if scope_parts else "All Posts Baseline"

        st.markdown(f"#### Format Comparison (Active Scope: {scope_label}, N={len(filtered_content)})")

        # Dynamically compute comparison on filtered_content
        cmp_res = compute_format_comparison(filtered_content)
        current_ab = cmp_res["data"]

        if cmp_res["status"] == "empty":
            st.info(cmp_res["message"])
        else:
            c_ab1, c_ab2 = st.columns([1.2, 1])

            with c_ab1:
                fig_ab = px.bar(
                    current_ab,
                    x="format",
                    y="average_engagement",
                    color="format",
                    color_discrete_map={
                        "Short": "#a855f7",
                        "Image": "#38bdf8",
                        "Carousel": "#06b6d4",
                        "Reel": "#6366f1"
                    },
                    text=current_ab["average_engagement"].apply(lambda v: f"{v:.2f}%"),
                    labels={"format": "Content Format", "average_engagement": "Average Engagement Rate (%)"},
                    title=f"Format Performance — {scope_label}"
                )
                apply_plotly_theme(fig_ab, show_legend=False)
                fig_ab.update_xaxes(showgrid=False)
                st.plotly_chart(fig_ab, use_container_width=True)

            with c_ab2:
                st.markdown("##### Format Metrics Breakdown")
                ab_display = current_ab.copy()
                ab_display["average_engagement"] = ab_display["average_engagement"].round(2).astype(str) + "%"
                ab_display["highest_engagement"] = ab_display["highest_engagement"].round(2).astype(str) + "%"
                ab_display["lowest_engagement"] = ab_display["lowest_engagement"].round(2).astype(str) + "%"
                st.dataframe(ab_display, use_container_width=True, hide_index=True)

                if cmp_res["status"] == "multi_format":
                    best_fmt = cmp_res["best_format"]
                    second_fmt = cmp_res["second_best"]
                    diff = cmp_res["difference"]

                    st.markdown(f"""
                    <div class="glass-card">
                        <span class="badge-caveat">Descriptive Difference: {diff:.2f} percentage points</span>
                        <div style="margin-top: 10px; font-size: 13px; color: #e2e8f0; line-height: 1.5;">
                            Highest observed format is <strong style="color: #c084fc;">{best_fmt['format']}</strong> ({best_fmt['average_engagement']:.2f}% avg, N={best_fmt['posts']}) 
                            compared to <strong style="color: #38bdf8;">{second_fmt['format']}</strong> ({second_fmt['average_engagement']:.2f}% avg, N={second_fmt['posts']}).
                        </div>
                        <div style="margin-top: 8px; font-size: 11px; color: #94a3b8;">
                            <em>Note: This observational difference reflects the selected sample ({scope_label}) and does not prove causal uplift.</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Single format selected or present
                    st.info(f"ℹ️ {cmp_res['message']}")

        # Baseline benchmark comparison for reference
        with st.expander("📊 View All-Dataset Baseline Benchmark (All Topics, N=50)"):
            if not ab_df.empty:
                st.caption("Reference benchmark across the complete 50-post dataset regardless of sidebar filters:")
                st.dataframe(ab_df, use_container_width=True, hide_index=True)
            else:
                st.info("A/B benchmark file not found.")

        st.write("")
        st.markdown("#### Performance Breakdown by Topic")
        topic_summary = (
            filtered_content.groupby("topic")["engagement_rate"]
            .agg(posts="count", avg_engagement="mean", min_rate="min", max_rate="max")
            .reset_index()
            .sort_values("avg_engagement", ascending=False)
        )

        fig_topic = px.bar(
            topic_summary,
            x="topic",
            y="avg_engagement",
            color="posts",
            color_continuous_scale=[[0, "#1e1b4b"], [0.5, "#7c3aed"], [1.0, "#38bdf8"]],
            text=topic_summary.apply(lambda r: f"{r['avg_engagement']:.2f}% (N={r['posts']})", axis=1),
            labels={"topic": "Topic", "avg_engagement": "Average Engagement Rate (%)", "posts": "Sample Posts (N)"},
            title="Topic Performance with Sample Size Support (N)"
        )
        apply_plotly_theme(fig_topic)
        st.plotly_chart(fig_topic, use_container_width=True)

# PAGE 3: AUDIENCE SENTIMENT INTELLIGENCE

elif nav_page == "Audience Sentiment Intelligence":
    st.markdown("""
    <div class="section-header-wrap">
        <div class="section-pill-dot"></div>
        <div class="section-title">Audience Sentiment & Comment Feedback Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    if comments_df.empty:
        st.warning("Audience comments dataset not found.")
    else:
        c_s1, c_s2 = st.columns([1, 1.6])

        with c_s1:
            st.markdown("#### Comment Label Distribution")
            label_counts = comments_df["sentiment_label"].value_counts().reset_index()
            label_counts.columns = ["sentiment_label", "count"]
            label_counts["percentage"] = (label_counts["count"] / len(comments_df) * 100).round(1)

            fig_sent = px.pie(
                label_counts,
                values="count",
                names="sentiment_label",
                color="sentiment_label",
                color_discrete_map={
                    "relatable": "#a855f7",
                    "neutral": "#38bdf8",
                    "unrelated": "#64748b"
                },
                hole=0.54,
                title=f"Audience Feedback Categories (Total N={len(comments_df)})"
            )
            apply_plotly_theme(fig_sent)
            fig_sent.update_layout(
                title=dict(
                    text=f"Audience Feedback Categories (Total N={len(comments_df)})",
                    x=0.5,
                    xanchor="center",
                    y=0.96,
                    yanchor="top",
                    font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=13, color="#f8fafc"),
                    pad=dict(b=12)
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.08,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(11, 15, 31, 0.75)",
                    bordercolor="rgba(139, 92, 246, 0.2)",
                    borderwidth=1,
                    font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#cbd5e1", size=11),
                    title=dict(text="")
                ),
                margin=dict(l=20, r=20, t=55, b=45)
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            st.caption(
                "**NLP Label Note:** The model categorizes comments into `relatable`, `neutral`, and `unrelated` "
                "based on the synthetic training corpus. Slang, dialects, and sarcastic nuances can introduce classification uncertainty."
            )

        with c_s2:
            st.markdown("#### Representative Comment Records")
            selected_label = st.selectbox(
                "Filter comments by category",
                ["All"] + sorted(comments_df["sentiment_label"].unique().tolist())
            )
            view_comments = comments_df.copy()
            if selected_label != "All":
                view_comments = view_comments[view_comments["sentiment_label"] == selected_label]

            st.dataframe(
                view_comments[["comment_id", "post_id", "comment_date", "sentiment_label", "comment_text"]],
                use_container_width=True,
                hide_index=True,
                height=420,
                column_config={
                    "comment_id": st.column_config.TextColumn(
                        "Comment ID",
                        help="Unique comment identifier",
                        width="small",
                    ),
                    "post_id": st.column_config.TextColumn(
                        "Post ID",
                        help="Associated post reference identifier",
                        width="small",
                    ),
                    "comment_date": st.column_config.TextColumn(
                        "Comment Date",
                        help="Date when comment was published",
                        width="medium",
                    ),
                    "sentiment_label": st.column_config.TextColumn(
                        "Sentiment Category",
                        help="Classified audience sentiment category",
                        width="medium",
                    ),
                    "comment_text": st.column_config.TextColumn(
                        "Comment Text",
                        help="Full text of the audience comment feedback",
                        width="large",
                    ),
                }
            )

# PAGE 4: VIRALITY PREDICTION & EXPLAINABLE AI

elif nav_page == "Virality Prediction & Explainable AI":
    st.markdown("""
    <div class="section-header-wrap">
        <div class="section-pill-dot"></div>
        <div class="section-title">🚀 Virality Prediction & Explainable AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Estimate whether a planned post is likely to achieve high engagement relative to the historical sample median.")

    model, meta = load_virality_model()

    if model is None:
        st.error("Virality model not found in `models/virality_model.pkl`. Please run `python src/virality_prediction.py`.")
    else:
        threshold = meta.get("threshold", 11.17)
        target_names = meta.get("target_names", {
            0: f"Standard / Moderate Engagement (< {threshold:.2f}%)",
            1: f"High Engagement Potential (>= {threshold:.2f}%)"
        })

        col_form, col_result = st.columns([1, 1.2])

        with col_form:
            st.markdown("#### Planned Post Attributes")
            with st.form("virality_form"):
                topic = st.selectbox("Content Topic", ["Lifestyle", "Comedy", "Gaming", "Education", "Motivation", "Friendship", "Food", "Technology"])
                content_format = st.selectbox("Content Format", ["Reel", "Short", "Image", "Carousel"])
                hook_type = st.selectbox("Hook Type", ["Question", "Story Opening", "Unexpected Visual", "Bold Statement", "Problem-Solution"])
                caption_style = st.selectbox("Caption Style", ["Story", "Funny", "Call to Action", "Informative", "Question"])
                posting_hour = st.slider("Posting Hour (Local Time)", 0, 23, 18, format="%d:00")
                submitted = st.form_submit_button("Run Prediction & Explanation", use_container_width=True)

        input_df = pd.DataFrame([{
            "topic": topic,
            "format": content_format,
            "hook_type": hook_type,
            "caption_style": caption_style,
            "posting_hour": posting_hour
        }])

        with col_result:
            st.markdown("#### Model Assessment & Class Mapping")

            # Prediction
            pred_class = int(model.predict(input_df)[0])
            mapped_label = target_names.get(pred_class, f"Class {pred_class}")

            # Confidence / Probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                classes = list(model.classes_)
                high_idx = classes.index(1) if 1 in classes else 1
                low_idx = classes.index(0) if 0 in classes else 0
                prob_high = float(probs[high_idx])
                prob_low = float(probs[low_idx])
            else:
                prob_high, prob_low = 0.5, 0.5

            if pred_class == 1:
                st.success(f"🎯 **Outcome:** {mapped_label}")
            else:
                st.info(f"📊 **Outcome:** {mapped_label}")

            st.write("")
            st.markdown(f"**Ensemble Confidence Breakdown:** (Sum: {prob_high+prob_low:.1f})")
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("High Engagement Vote", f"{prob_high:.1%}")
            p_col2.metric("Standard/Moderate Vote", f"{prob_low:.1%}")

            st.progress(prob_high)
            st.caption(
                f"**Threshold Definition:** The model defines 'High Engagement' as meeting or exceeding {threshold:.2f}% "
                f"engagement rate (the historical sample median). "
                f"Confidence reflects Random Forest tree voting agreement (sample N=50), not calibrated real-world virality."
            )

        # Explainable AI: Feature Contributions
        st.write("")
        st.markdown("---")
        st.markdown("#### 🔍 Explainable AI: Local Feature Contributions")
        st.caption("How each input attribute shifts the estimated probability relative to the baseline training population:")

        contribs = compute_feature_contributions(model, input_df, content_df)
        if contribs:
            contrib_rows = []
            for feat, info in contribs.items():
                contrib_rows.append({
                    "Feature": feat.replace("_", " ").title(),
                    "Selected Value": str(info["value"]),
                    "Baseline Default": str(info["baseline_value"]),
                    "Probability Shift": f"{info['delta_probability']:+.1%}",
                    "Impact Direction": "Positive (+)" if info["delta_probability"] > 0 else ("Negative (-)" if info["delta_probability"] < 0 else "Neutral")
                })
            st.dataframe(pd.DataFrame(contrib_rows), use_container_width=True, hide_index=True)

        with st.expander("ℹ️ Model Architecture & Validation Card"):
            st.markdown(f"""
            - **Algorithm:** Scikit-learn Pipeline (OneHotEncoder + RandomForestClassifier, 100 trees, balanced weights)
            - **Training Samples:** 40 posts | **Test Split:** 10 posts
            - **Held-out Split Accuracy:** {meta.get('test_accuracy', 0.60):.1%}
            - **5-Fold Stratified CV Mean:** {meta.get('cv_mean_accuracy', 0.50):.1%} *(Reflects honest generalization baseline on small sample)*
            - **Guardrail:** Observational associations from synthetic data. Do not assume causal guarantees.
            """)

# PAGE 5: WHAT-IF CONTENT SIMULATOR

elif nav_page == "What-If Content Simulator":
    st.markdown("""
    <div class="section-header-wrap">
        <div class="section-pill-dot"></div>
        <div class="section-title">🧪 What-If Content Simulator</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Simulate adjustments to a post's format, topic, hook, or posting hour, and compare the baseline vs. scenario estimates side-by-side.")

    model, meta = load_virality_model()
    if model is None:
        st.error("Model unavailable. Train model first via `src/virality_prediction.py`.")
    else:
        threshold = meta.get("threshold", 11.17)

        c_base, c_scen = st.columns(2)

        with c_base:
            st.markdown("##### 📌 Baseline Configuration")
            b_topic = st.selectbox("Baseline Topic", ["Lifestyle", "Comedy", "Gaming", "Education", "Motivation", "Friendship", "Food", "Technology"], index=1, key="b_topic")
            b_format = st.selectbox("Baseline Format", ["Reel", "Short", "Image", "Carousel"], index=0, key="b_format")
            b_hook = st.selectbox("Baseline Hook", ["Question", "Story Opening", "Unexpected Visual", "Bold Statement", "Problem-Solution"], index=0, key="b_hook")
            b_caption = st.selectbox("Baseline Caption", ["Story", "Funny", "Call to Action", "Informative", "Question"], index=1, key="b_caption")
            b_hour = st.slider("Baseline Posting Hour", 0, 23, 13, key="b_hour")

        with c_scen:
            st.markdown("##### ⚡ Proposed Scenario Variant")
            s_topic = st.selectbox("Scenario Topic", ["Lifestyle", "Comedy", "Gaming", "Education", "Motivation", "Friendship", "Food", "Technology"], index=1, key="s_topic")
            s_format = st.selectbox("Scenario Format", ["Reel", "Short", "Image", "Carousel"], index=1, key="s_format")
            s_hook = st.selectbox("Scenario Hook", ["Question", "Story Opening", "Unexpected Visual", "Bold Statement", "Problem-Solution"], index=2, key="s_hook")
            s_caption = st.selectbox("Scenario Caption", ["Story", "Funny", "Call to Action", "Informative", "Question"], index=1, key="s_caption")
            s_hour = st.slider("Scenario Posting Hour", 0, 23, 18, key="s_hour")

        df_b = pd.DataFrame([{"topic": b_topic, "format": b_format, "hook_type": b_hook, "caption_style": b_caption, "posting_hour": b_hour}])
        df_s = pd.DataFrame([{"topic": s_topic, "format": s_format, "hook_type": s_hook, "caption_style": s_caption, "posting_hour": s_hour}])

        prob_b = float(model.predict_proba(df_b)[0][1])
        prob_s = float(model.predict_proba(df_s)[0][1])
        diff_prob = prob_s - prob_b

        st.write("")
        st.markdown("#### Scenario Comparison Results")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.markdown(
                render_kpi_card(
                    label="Baseline Confidence",
                    value=f"{prob_b:.1%}",
                    subtext="High Engagement" if prob_b >= 0.5 else "Standard / Moderate",
                    icon="📌",
                    color="#6366f1",
                    trend="stable"
                ),
                unsafe_allow_html=True
            )
        with res_col2:
            st.markdown(
                render_kpi_card(
                    label="Scenario Confidence",
                    value=f"{prob_s:.1%}",
                    subtext="High Engagement" if prob_s >= 0.5 else "Standard / Moderate",
                    icon="⚡",
                    color="#38bdf8",
                    trend="growth" if prob_s >= prob_b else "down"
                ),
                unsafe_allow_html=True
            )
        with res_col3:
            delta_color = "#34d399" if diff_prob > 0 else ("#f43f5e" if diff_prob < 0 else "#94a3b8")
            st.markdown(
                render_kpi_card(
                    label="Confidence Delta",
                    value=f"{diff_prob:+.1%}",
                    subtext="Estimated model probability shift",
                    icon="📊",
                    color=delta_color,
                    trend="up" if diff_prob > 0 else "down"
                ),
                unsafe_allow_html=True
            )

        # Highlight changes
        changes = []
        if b_topic != s_topic: changes.append(f"Topic changed from `{b_topic}` to `{s_topic}`")
        if b_format != s_format: changes.append(f"Format changed from `{b_format}` to `{s_format}`")
        if b_hook != s_hook: changes.append(f"Hook changed from `{b_hook}` to `{s_hook}`")
        if b_caption != s_caption: changes.append(f"Caption changed from `{b_caption}` to `{s_caption}`")
        if b_hour != s_hour: changes.append(f"Posting hour changed from `{b_hour:02d}:00` to `{s_hour:02d}:00`")

        if changes:
            st.markdown("**Simulated Parameter Deltas:**")
            for c in changes:
                st.markdown(f"- {c}")
        else:
            st.info("Baseline and scenario parameters are currently identical.")

        st.caption("⚠️ **Simulator Disclaimer:** These outputs represent statistical correlations captured by the trained model. They do not constitute guaranteed causal lift.")

# PAGE 6: CONTENT DNA ANALYZER

elif nav_page == "Content DNA Analyzer":
    st.markdown("""
    <div class="section-header-wrap">
        <div class="section-pill-dot"></div>
        <div class="section-title">🧬 Content DNA Analyzer</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Examine multi-dimensional engagement profiles across format, topic, hook type, and posting hour, with explicit sample suppression for small groups.")

    if content_df.empty:
        st.warning("Content data unavailable.")
    else:
        min_sample = st.slider("Minimum Sample Threshold (N) to qualify findings", min_value=1, max_value=10, value=3)

        dna = compute_content_dna(content_df, min_sample=min_sample)

        tab_matrix, tab_dims = st.tabs(["Topic × Format Matrix Heatmap", "Attribute DNA Breakdowns"])

        with tab_matrix:
            st.markdown("#### Average Engagement Rate Matrix: Topic vs. Format")
            pivot_mean = dna["topic_format_matrix"]
            pivot_count = dna["topic_format_counts"]

            # Heatmap with Midnight Navy to Electric Blue colorscale
            fig_heat = px.imshow(
                pivot_mean,
                labels=dict(x="Content Format", y="Topic", color="Avg Engagement (%)"),
                x=pivot_mean.columns,
                y=pivot_mean.index,
                color_continuous_scale=[[0, "#060813"], [0.35, "#312e81"], [0.7, "#7c3aed"], [1.0, "#38bdf8"]],
                text_auto=".2f",
                aspect="auto"
            )
            apply_plotly_theme(fig_heat, show_legend=False)
            st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("##### Supporting Post Counts (N) Matrix")
            st.dataframe(pivot_count, use_container_width=True)
            st.caption(
                "Notice cells with N=1 or N=0. High averages in low-N cells (e.g. Lifestyle Reel N=1, Carousel N=1) "
                "must not be interpreted as statistically solid conclusions."
            )

        with tab_dims:
            selected_dim = st.selectbox(
                "Select Dimension to Inspect",
                ["format", "topic", "hook_type", "caption_style", "posting_hour"]
            )
            dim_df = dna["dimensions"][selected_dim]

            fig_dim = px.bar(
                dim_df,
                x=selected_dim,
                y="avg_engagement",
                color="is_sufficient_sample",
                color_discrete_map={True: "#38bdf8", False: "#f59e0b"},
                text=dim_df.apply(lambda r: f"{r['avg_engagement']:.2f}% (N={r['posts']})", axis=1),
                labels={selected_dim: selected_dim.replace("_", " ").title(), "avg_engagement": "Average Engagement Rate (%)", "is_sufficient_sample": f"Sample N >= {min_sample}"},
                title=f"Engagement by {selected_dim.replace('_', ' ').title()} with Sample Qualification"
            )
            apply_plotly_theme(fig_dim)
            st.plotly_chart(fig_dim, use_container_width=True)

            st.dataframe(dim_df, use_container_width=True, hide_index=True)

# PAGE 7: ENGAGEMENT ANOMALY DETECTION

elif nav_page == "Engagement Anomaly Detection":
    st.markdown("""
    <div class="section-header-wrap">
        <div class="section-pill-dot"></div>
        <div class="section-title">🚨 Engagement Anomaly Detection</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Detect unusual engagement behavior using robust statistical bounds (Interquartile Range) to isolate genuine viral spikes from data entry artifacts.")

    if content_df.empty:
        st.warning("Content data unavailable.")
    else:
        anom_df, meta = detect_anomalies(content_df)

        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        with a_col1:
            st.markdown(
                render_kpi_card(
                    label="IQR Lower Bound",
                    value=f"{meta['lower_bound']:.2f}%",
                    subtext="Q1 - 1.5 * IQR cutoff",
                    icon="📉",
                    color="#f43f5e",
                    trend="down"
                ),
                unsafe_allow_html=True
            )
        with a_col2:
            st.markdown(
                render_kpi_card(
                    label="Median Engagement",
                    value=f"{content_df['engagement_rate'].median():.2f}%",
                    subtext="Historical dataset midpoint",
                    icon="⚖️",
                    color="#38bdf8",
                    trend="stable"
                ),
                unsafe_allow_html=True
            )
        with a_col3:
            st.markdown(
                render_kpi_card(
                    label="IQR Upper Threshold",
                    value=f"{meta['upper_bound']:.2f}%",
                    subtext="Q3 + 1.5 * IQR cutoff",
                    icon="📈",
                    color="#a855f7",
                    trend="growth"
                ),
                unsafe_allow_html=True
            )
        with a_col4:
            st.markdown(
                render_kpi_card(
                    label="Flagged High Spikes",
                    value=f"{meta['spike_count']} posts",
                    subtext="Outlier viral events identified",
                    icon="⚡",
                    color="#34d399",
                    trend="up"
                ),
                unsafe_allow_html=True
            )

        st.write("")

        # Scatter plot highlighting anomalies
        anom_df["post_date"] = pd.to_datetime(anom_df["post_date"])
        fig_anom = px.scatter(
            anom_df,
            x="post_date",
            y="engagement_rate",
            color="anomaly_type",
            color_discrete_map={"Normal": "#38bdf8", "High Spike": "#a855f7", "Low Dip": "#f43f5e"},
            size="views",
            hover_data=["post_id", "topic", "format", "total_interactions", "anomaly_details"],
            title="Post Engagement Distribution with Statistical IQR Boundaries"
        )
        fig_anom.add_hline(y=meta["upper_bound"], line_dash="dash", line_color="#a855f7", annotation_text=f"Upper Bound ({meta['upper_bound']:.2f}%)")
        fig_anom.add_hline(y=meta["lower_bound"], line_dash="dash", line_color="#f43f5e", annotation_text=f"Lower Bound ({meta['lower_bound']:.2f}%)")

        apply_plotly_theme(fig_anom)
        st.plotly_chart(fig_anom, use_container_width=True)

        st.markdown("#### Audit Table of Flagged Posts")
        spikes = anom_df[anom_df["is_anomaly"]]
        if spikes.empty:
            st.info("No posts exceed the statistical IQR bounds in the current dataset.")
        else:
            st.dataframe(
                spikes[["post_id", "post_date", "topic", "format", "views", "engagement_rate", "anomaly_type", "anomaly_details"]],
                use_container_width=True,
                hide_index=True
            )

# PAGE 8: TREND FORECASTING & BASELINES

elif nav_page == "Trend Forecasting & Baselines":
    st.markdown("""
    <div class="section-header-wrap">
        <div class="section-pill-dot"></div>
        <div class="section-title">📉 Engagement Trend Forecasting & Baselines</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Interactive time series tracking daily observed engagement, a 3-period smoothing rolling average, and a linear trend projection compared against naive baselines.")

    if trend_df.empty:
        st.info("Trend forecast dataset not found. Run `python src/trend_forecasting.py`.")
    else:
        trend_plot = trend_df.copy()
        trend_plot["post_date"] = pd.to_datetime(trend_plot["post_date"])

        # Interactive Plotly Chart with Aurora Purple & Electric Blue Aesthetics
        fig_trend = go.Figure()

        # Observed points
        fig_trend.add_trace(go.Scatter(
            x=trend_plot["post_date"],
            y=trend_plot["engagement_rate"],
            mode="lines+markers",
            name="Daily Observed Engagement (%)",
            line=dict(color="#38bdf8", width=1.8),
            marker=dict(size=6, color="#0ea5e9", line=dict(width=1, color="#38bdf8"))
        ))

        # Rolling average
        fig_trend.add_trace(go.Scatter(
            x=trend_plot["post_date"],
            y=trend_plot["rolling_average"],
            mode="lines",
            name="3-Period Rolling Average (%)",
            line=dict(color="#a855f7", width=2.8)
        ))

        # Linear trend estimate
        fig_trend.add_trace(go.Scatter(
            x=trend_plot["post_date"],
            y=trend_plot["trend_estimate"],
            mode="lines",
            name="Linear Projection Line (%)",
            line=dict(color="#f59e0b", width=2, dash="dash")
        ))

        fig_trend.update_layout(
            title="Social Engagement Rate Over Time (%)",
            xaxis_title="Post Date (YYYY-MM-DD)",
            yaxis_title="Engagement Rate (%)"
        )
        apply_plotly_theme(fig_trend)
        st.plotly_chart(fig_trend, use_container_width=True)

        # Baseline evaluation metrics
        actual = trend_plot["engagement_rate"].values
        trend_pred = trend_plot["trend_estimate"].values
        mean_base = np.full_like(actual, actual.mean())
        naive_base = np.roll(actual, 1)
        naive_base[0] = actual[0]

        t_mae = np.mean(np.abs(actual - trend_pred))
        n_mae = np.mean(np.abs(actual - naive_base))
        m_mae = np.mean(np.abs(actual - mean_base))

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(
                render_kpi_card(
                    label="Latest Rolling Avg",
                    value=f"{trend_plot['rolling_average'].iloc[-1]:.2f}%",
                    subtext="3-period smoothed rate",
                    icon="📊",
                    color="#a855f7",
                    trend="stable"
                ),
                unsafe_allow_html=True
            )
        with b2:
            st.markdown(
                render_kpi_card(
                    label="Linear Model MAE",
                    value=f"{t_mae:.2f}%",
                    subtext="Mean absolute projection error",
                    icon="📐",
                    color="#38bdf8",
                    trend="stable"
                ),
                unsafe_allow_html=True
            )
        with b3:
            st.markdown(
                render_kpi_card(
                    label="Last-Value Naive MAE",
                    value=f"{n_mae:.2f}%",
                    subtext="Shift-1 benchmark baseline",
                    icon="⏱️",
                    color="#6366f1",
                    trend="stable"
                ),
                unsafe_allow_html=True
            )
        with b4:
            st.markdown(
                render_kpi_card(
                    label="Historical Mean MAE",
                    value=f"{m_mae:.2f}%",
                    subtext="Overall mean baseline error",
                    icon="🎯",
                    color="#22d3ee",
                    trend="stable"
                ),
                unsafe_allow_html=True
            )

        st.caption(
            "**Forecast Evaluation:** The linear trend error (MAE 2.59%) performs comparably to the historical mean baseline (MAE 2.58%). "
            "Because social algorithms and audience dynamics shift rapidly, trend projections are descriptive benchmarks, not guarantees."
        )

# PAGE 9: STRATEGY & EXPERIMENT PLANNER

elif nav_page == "Strategy & Experiment Planner":
    st.markdown("""
    <div class="section-header-wrap">
        <div class="section-pill-dot"></div>
        <div class="section-title">🎯 Evidence-Linked Strategy & Experiment Planner</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Convert observed engagement patterns into measurable publishing experiments. Track hypotheses, target metrics, and outcomes without auto-posting.")

    # Generate evidence-linked recommendations
    rec_data = generate_recommendations(content_df)

    if rec_data:
        st.markdown("#### 1. Sample-Derived Recommendations & Caveats")
        for rec in rec_data["recommendations"]:
            st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <span style="font-weight: 700; font-size: 16px; color: #c084fc;">[{rec['dimension']}] Recommend: {rec['recommendation']}</span>
                    <span class="{'badge-verified' if 'Candidate' in rec['status'] else 'badge-caveat'}">{rec['status']} (N={rec['sample_count']})</span>
                </div>
                <div style="margin-top: 10px; font-size: 13px; color: #e2e8f0;">
                    Observed Average: <strong style="color: #38bdf8;">{rec['average_engagement']:.2f}%</strong> (Dataset Baseline: {rec['baseline_engagement']:.2f}%)
                </div>
                <div style="margin-top: 6px; font-size: 12px; color: #94a3b8; line-height: 1.4;">
                    <strong>Evidence & Caveat:</strong> {rec['caveat']}
                </div>
                <div style="margin-top: 6px; font-size: 12px; color: #38bdf8; line-height: 1.4;">
                    <strong>Suggested Experiment:</strong> {rec['suggested_experiment']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### 2. Plan a Controlled Creator Experiment")

    with st.form("experiment_form"):
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            hypothesis = st.text_input("Experiment Hypothesis", "Testing Short vs Carousel in Motivation topic with Question hooks")
            test_format = st.selectbox("Variant Format", ["Short", "Carousel", "Reel", "Image"])
            test_topic = st.selectbox("Variant Topic", ["Motivation", "Comedy", "Gaming", "Education", "Lifestyle", "Friendship", "Food", "Technology"])
        with col_e2:
            primary_metric = st.selectbox("Primary Success Metric", ["Engagement Rate (%)", "Retention Rate (%)", "Shares & Saves Ratio"])
            target_sample_size = st.number_input("Target Post Sample Size per variant", min_value=3, max_value=20, value=5)
            scheduled_window = st.text_input("Planned Posting Window & Timezone", "18:00 - 19:00 IST")

        submitted_exp = st.form_submit_button("Save Experiment Plan", use_container_width=True)

    if submitted_exp:
        current_experiments = load_experiments()
        new_exp = {
            "experiment_id": f"EXP-{len(current_experiments)+1:03d}",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "hypothesis": hypothesis,
            "format": test_format,
            "topic": test_topic,
            "primary_metric": primary_metric,
            "target_sample_size": int(target_sample_size),
            "scheduled_window": scheduled_window,
            "status": "Planned"
        }
        current_experiments.append(new_exp)
        save_experiments(current_experiments)
        st.success(f"Experiment {new_exp['experiment_id']} registered successfully!")

    # Experiment Log
    st.write("")
    st.markdown("#### 3. Active & Completed Experiment Log")
    experiments = load_experiments()
    if experiments:
        exp_df = pd.DataFrame(experiments)
        st.dataframe(exp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No experiments planned yet. Use the form above to record your first experiment.")

# 7. FOOTER

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 12px; padding: 14px 0; line-height: 1.6;'>"
    "<strong style='color: #94a3b8;'>CreatorPulse AI • Commercial Analytics Platform Demo</strong><br/>"
    "Academic Demonstration Environment • Synthetic Dataset (50 posts, 150 comments)<br/>"
    "Models, predictions, and recommendations are exploratory estimates and do not guarantee future social reach."
    "</div>",
    unsafe_allow_html=True
)
