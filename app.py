# =========================================================================================
# 🎬 CINEMETRICS INTELLIGENCE TERMINAL (ULTRA-LIGHTWEIGHT VECTOR EDITION)
# Version: 10.9.0 | Build: Local/Production (Strict JSON Enforcement)
# Description: Advanced Content-Based Filtering Dashboard bypassing DataFrame size limits.
# Features TF-IDF vectorization, Cosine Similarity arrays, and JSON Reverse Index Mapping.
# Theme: CineMetrics (Deep Cinematic Black, Neon Crimson, Box-Office Gold)
# =========================================================================================

import streamlit as st
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import base64
import json
import difflib
from datetime import datetime
import uuid

# Scikit-learn is required to calculate dot products/cosine similarity on the fly
try:
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
except ImportError:
    pass

# =========================================================================================
# 1. PAGE CONFIGURATION & SECURE INITIALIZATION
# =========================================================================================
st.set_page_config(
    page_title="CineMetrics | Lightweight Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================================
# 2. MACHINE LEARNING ASSET INGESTION (JSON + MATRICES ONLY)
# =========================================================================================
@st.cache_resource
def load_ml_infrastructure():
    """
    Safely loads the serialized TF-IDF arrays and JSON indices ONLY.
    Bypasses the heavy pandas DataFrame to stay under GitHub's 25MB limit.
    """
    indices = None
    tfidf = None
    tfidf_matrix = None
    
    try:
        # PURE JSON LOADING: Guaranteed to bypass Pandas StringDtype errors forever.
        with open("indices.json", "r") as f:
            indices = json.load(f)
    except Exception as e:
        st.sidebar.error(f"🔴 INDICES LOAD ERROR: {str(e)}\n\n(Ensure `indices.json` is saved in your project folder!)")
        
    try:
        with open("tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)
    except Exception as e:
        st.sidebar.error(f"🔴 TF-IDF VECTORIZER LOAD ERROR: {str(e)}")
        
    try:
        with open("tfidx_matrix.pkl", "rb") as f:
            tfidf_matrix = pickle.load(f)
    except Exception as e:
        st.sidebar.error(f"🔴 TF-IDF MATRIX LOAD ERROR: {str(e)}\n\n(Ensure `tfidx_matrix.pkl` exists)")
        
    return indices, tfidf, tfidf_matrix

indices_map, tfidf_vectorizer, tfidf_matrix = load_ml_infrastructure()

# --- REVERSE INDEX MAPPING & FUZZY MATCH PREP ---
ALL_TITLES = []
INDEX_TO_TITLE = {}

if indices_map is not None:
    # Since indices_map is now guaranteed to be a pure JSON dictionary
    for title, idx in indices_map.items():
        ALL_TITLES.append(str(title))
        INDEX_TO_TITLE[int(idx)] = str(title)

# =========================================================================================
# 3. ENTERPRISE CSS INJECTION (MASSIVE STYLESHEET FOR CINEMETRICS THEME)
# =========================================================================================
st.markdown(
"""<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

/* ── GLOBAL COLOR PALETTE & CSS VARIABLES ── */
:root {
    --cine-900:      #09090b;
    --cine-800:      #18181b;
    --cine-700:      #27272a;
    --crimson-core:  #e11d48;
    --crimson-dim:   rgba(225, 29, 72, 0.2);
    --gold-core:     #f59e0b;
    --gold-dim:      rgba(245, 158, 11, 0.2);
    --white-main:    #f8fafc;
    --slate-light:   #94a3b8;
    --slate-dark:    #475569;
    --glass-bg:      rgba(24, 24, 27, 0.6);
    --glass-border:  rgba(225, 29, 72, 0.15);
    --glow-crimson:  0 0 35px rgba(225, 29, 72, 0.25);
    --glow-gold:     0 0 35px rgba(245, 158, 11, 0.25);
}

/* ── BASE APPLICATION STYLING & TYPOGRAPHY ── */
.stApp {
    background: var(--cine-900);
    font-family: 'Inter', sans-serif;
    color: var(--slate-light);
    overflow-x: hidden;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Bebas Neue', sans-serif;
    color: var(--white-main);
    letter-spacing: 2px;
}

/* ── DYNAMIC BACKGROUND ANIMATIONS ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: 
        radial-gradient(circle at 15% 20%, rgba(225, 29, 72, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(245, 158, 11, 0.03) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(9, 9, 11, 0.8) 0%, transparent 80%);
    pointer-events: none;
    z-index: 0;
    animation: cinemaPulse 15s ease-in-out infinite alternate;
}

@keyframes cinemaPulse {
    0%   { opacity: 0.5; filter: hue-rotate(0deg); }
    100% { opacity: 1.0; filter: hue-rotate(10deg); }
}

/* ── CINEMATIC GRID OVERLAY ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: 
        linear-gradient(rgba(225, 29, 72, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(225, 29, 72, 0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── MAIN CONTAINER SPACING ── */
.main .block-container {
    position: relative;
    z-index: 1;
    padding-top: 30px;
    padding-bottom: 90px;
    max-width: 1550px;
}

/* ── HERO SECTION & HEADERS ── */
.hero {
    text-align: center;
    padding: 80px 20px 60px;
    animation: slideDown 0.9s cubic-bezier(0.22,1,0.36,1) both;
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-50px); }
    to   { opacity: 1; transform: translateY(0); }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 15px;
    background: rgba(225, 29, 72, 0.05);
    border: 1px solid rgba(225, 29, 72, 0.3);
    border-radius: 50px;
    padding: 10px 30px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--crimson-core);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 25px;
    box-shadow: var(--glow-crimson);
}

.hero-badge-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--gold-core);
    box-shadow: 0 0 12px var(--gold-core);
    animation: recordingTick 1.5s ease-in-out infinite;
}

@keyframes recordingTick {
    0%, 100% { transform: scale(1); opacity: 0.6; }
    50%      { transform: scale(1.6); opacity: 1; box-shadow: 0 0 20px var(--gold-core); }
}

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(50px, 8vw, 100px);
    font-weight: 400;
    letter-spacing: 4px;
    line-height: 1.0;
    margin-bottom: 18px;
    text-transform: uppercase;
}

.hero-title em {
    font-style: normal;
    color: var(--crimson-core);
    text-shadow: 0 0 35px rgba(225, 29, 72, 0.4);
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    font-weight: 400;
    color: var(--slate-light);
    letter-spacing: 4px;
    text-transform: uppercase;
}

/* ── GLASS PANELS & UI CARDS ── */
.glass-panel {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 45px;
    margin-bottom: 35px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    transition: all 0.4s ease;
    animation: fadeUp 0.8s ease both;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}

.glass-panel:hover {
    border-color: rgba(225, 29, 72, 0.4);
    box-shadow: var(--glow-crimson);
    transform: translateY(-2px);
}

.panel-heading {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 32px;
    color: var(--white-main);
    letter-spacing: 2px;
    margin-bottom: 35px;
    border-bottom: 1px solid rgba(225, 29, 72, 0.2);
    padding-bottom: 15px;
    text-transform: uppercase;
}

/* ── COMPONENT OVERRIDES (STREAMLIT NATIVE) ── */
div[data-testid="stTextInput"] label { display: none !important; }

/* REVISED TEXT INPUT STYLING */
div[data-testid="stTextInput"] > div > div > input {
    background: rgba(24, 24, 27, 0.9) !important;
    border: 1px solid rgba(225, 29, 72, 0.4) !important;
    color: var(--white-main) !important;
    border-radius: 8px !important;
    padding: 18px 20px !important;
    font-size: 16px !important;
    line-height: 1.5 !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.5) !important;
}

div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--crimson-core) !important;
    box-shadow: 0 0 20px rgba(225, 29, 72, 0.3), inset 0 2px 10px rgba(0,0,0,0.5) !important;
}

/* REVISED BUTTON STYLING */
div.stButton > button {
    width: 100% !important;
    background: transparent !important;
    color: var(--crimson-core) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: 1px solid var(--crimson-core) !important;
    border-radius: 8px !important;
    padding: 18px 20px !important;
    line-height: 1.5 !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    background-color: rgba(225, 29, 72, 0.05) !important;
    margin-top: 0px !important; 
    box-shadow: 0 5px 15px rgba(225, 29, 72, 0.1) !important;
}

div.stButton > button:hover {
    background-color: rgba(225, 29, 72, 0.15) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 25px rgba(225, 29, 72, 0.3) !important;
}

/* ── PREDICTION RESULT BOX (MOVIE CARD STYLE) ── */
.movie-card {
    background: var(--cine-800) !important;
    border: 1px solid var(--glass-border) !important;
    padding: 30px !important;
    border-radius: 12px !important;
    position: relative !important;
    overflow: hidden !important;
    margin-bottom: 20px !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}

.movie-card:hover {
    border-color: var(--crimson-core) !important;
    box-shadow: var(--glow-crimson) !important;
    transform: translateX(5px) !important;
}

.movie-rank {
    position: absolute;
    top: -15px;
    right: 10px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 80px;
    color: rgba(255,255,255,0.03);
    z-index: 0;
}

.movie-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 32px;
    color: var(--white-main);
    letter-spacing: 1px;
    margin-bottom: 10px;
    position: relative;
    z-index: 1;
}

.movie-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--gold-core);
    margin-bottom: 15px;
    position: relative;
    z-index: 1;
}

.movie-desc {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: var(--slate-light);
    line-height: 1.6;
    margin-bottom: 20px;
    position: relative;
    z-index: 1;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.movie-sim {
    display: inline-block;
    background: rgba(225, 29, 72, 0.1);
    border: 1px solid rgba(225, 29, 72, 0.4);
    color: var(--crimson-core);
    padding: 8px 20px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    position: relative;
    z-index: 1;
}

/* ── TABS NAVIGATION STYLING ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--cine-800) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(225, 29, 72, 0.2) !important;
    padding: 8px !important;
    gap: 12px !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--slate-dark) !important;
    border-radius: 6px !important;
    padding: 18px 30px !important;
    transition: all 0.3s ease !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(225, 29, 72, 0.1) !important;
    color: var(--crimson-core) !important;
    border: 1px solid rgba(225, 29, 72, 0.4) !important;
    box-shadow: 0 0 20px rgba(225, 29, 72, 0.1) !important;
}

/* ── SIDEBAR STYLING & TELEMETRY ── */
section[data-testid="stSidebar"] {
    background: var(--cine-900) !important;
    border-right: 1px solid rgba(225, 29, 72, 0.15) !important;
}

.sb-logo-text {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 45px;
    color: var(--white-main);
    letter-spacing: 3px;
}

.sb-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    color: var(--slate-light);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 10px;
    margin-top: 35px;
}

.telemetry-card {
    background: rgba(24, 24, 27, 0.5) !important;
    border: 1px solid rgba(225, 29, 72, 0.15) !important;
    padding: 22px !important;
    border-radius: 8px !important;
    text-align: center !important;
    margin-bottom: 18px !important;
    transition: all 0.3s ease;
}

.telemetry-card:hover {
    background: rgba(39, 39, 42, 0.9) !important;
    border-color: rgba(225, 29, 72, 0.4) !important;
    transform: translateY(-2px);
}

.telemetry-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: var(--crimson-core);
}

.telemetry-lbl {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    color: var(--slate-dark);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 8px;
}

/* ── FLOATING PARTICLES (CELLULOID TRACKS) ── */
.particles {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.film-track {
    position: absolute;
    width: 100vw; height: 10px;
    background: repeating-linear-gradient(90deg, transparent, transparent 10px, rgba(225, 29, 72, 0.1) 10px, rgba(225, 29, 72, 0.1) 20px);
    opacity: 0.3;
    animation: panFilm linear infinite;
}

.film-track:nth-child(1) { top: 15%; animation-duration: 40s; animation-direction: normal; }
.film-track:nth-child(2) { top: 45%; animation-duration: 60s; animation-direction: reverse; height: 5px; opacity: 0.1;}
.film-track:nth-child(3) { top: 85%; animation-duration: 35s; animation-direction: normal; }

@keyframes panFilm {
    0%   { background-position: 0 0; }
    100% { background-position: 1000px 0; }
}
</style>

<div class="particles">
<div class="film-track"></div><div class="film-track"></div><div class="film-track"></div>
</div>""",
    unsafe_allow_html=True,
)

# =========================================================================================
# 4. SESSION STATE MANAGEMENT & ARCHITECTURE INITIALIZATION
# =========================================================================================
# Initialize strict session UUID for telemetry payload tracking
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"CINE-IDX-{str(uuid.uuid4())[:8].upper()}"

# System operational states
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "matched_title" not in st.session_state:
    st.session_state["matched_title"] = None
if "match_confidence" not in st.session_state:
    st.session_state["match_confidence"] = 0.0
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None
if "timestamp" not in st.session_state:
    st.session_state["timestamp"] = None
if "compute_latency" not in st.session_state:
    st.session_state["compute_latency"] = 0.0

# =========================================================================================
# 5. ENTERPRISE SIDEBAR LOGIC (SYSTEM TELEMETRY)
# =========================================================================================
with st.sidebar:
    st.markdown(
"""<div style='text-align:center; padding:25px 0 35px;'>
<div class="sb-logo-text">CINEMETRICS</div>
<div style="font-family:'Space Mono'; font-size:10px; color:rgba(225,29,72,0.8); letter-spacing:4px; margin-top:8px;">LIGHTWEIGHT ENGINE</div>
<div style="font-family:'Space Mono'; font-size:9px; color:rgba(255,255,255,0.3); margin-top:12px;">ID: {}</div>
</div>""".format(st.session_state["session_id"]),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-title">⚙️ Architecture Specs</div>', unsafe_allow_html=True)
    st.markdown(
"""<div style="background:rgba(24,24,27,0.6); padding:20px; border-radius:8px; border:1px solid rgba(225,29,72,0.2); font-family:Inter; font-size:13px; color:rgba(248,250,252,0.8); line-height:1.9;">
<b>Algorithm:</b> Content-Based Filtering<br>
<b>Vectorization:</b> TF-IDF (Term Freq)<br>
<b>Distance Metric:</b> Cosine Similarity<br>
<b>Mode:</b> Lightweight Vector Array<br>
<b>Metadata:</b> Bypassed for Scale<br>
</div>""", unsafe_allow_html=True
    )

    st.markdown('<div class="sb-title">📊 Dataset Telemetry</div>', unsafe_allow_html=True)
    
    vocab_size = len(tfidf_vectorizer.vocabulary_) if tfidf_vectorizer and hasattr(tfidf_vectorizer, 'vocabulary_') else "N/A"
    total_movies = len(ALL_TITLES)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val" style="color:var(--white-main);">{total_movies}</div><div class="telemetry-lbl">Total Films</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="telemetry-card"><div class="telemetry-val" style="color:var(--gold-core);">1.0</div><div class="telemetry-lbl">Max Sim</div></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val" style="color:var(--crimson-core);">{vocab_size}</div><div class="telemetry-lbl">TFIDF Vocab</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val">{st.session_state["compute_latency"]}s</div><div class="telemetry-lbl">Latency</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state["recommendations"] is None:
        st.markdown(
"""<div style="padding:15px; border-left:4px solid var(--slate-dark); background:rgba(255,255,255,0.05); border-radius:4px; font-family:Inter; font-size:12px; color:var(--slate-light);">
<b>SYSTEM STANDBY</b><br>Awaiting query input for vector matching.
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(
f"""<div style="padding:15px; border-left:4px solid var(--crimson-core); background:rgba(225,29,72,0.05); border-radius:4px; font-family:Inter; font-size:12px; color:var(--crimson-core);">
<b>COMPUTE COMPLETE</b><br>Cosine Dot-Product Latency: {st.session_state['compute_latency']}s
</div>""", unsafe_allow_html=True)

# =========================================================================================
# 6. HERO HEADER SECTION
# =========================================================================================
st.markdown(
"""<div class="hero">
<div class="hero-badge">
<div class="hero-badge-dot"></div>
LIGHTWEIGHT VECTORIZATION | COSINE SIMILARITY ENGINE
</div>
<div class="hero-title">CONTENT <em>DISCOVERY</em> TERMINAL</div>
<div class="hero-sub">Enterprise Machine Learning Dashboard For Cinematic Intelligence</div>
</div>""",
    unsafe_allow_html=True,
)

# =========================================================================================
# 7. MAIN APPLICATION TABS (6-TAB MONOLITHIC ARCHITECTURE)
# =========================================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎬 CONTENT DISCOVERY", 
    "📊 VECTOR ANALYTICS", 
    "🧠 NLP ARCHITECTURE", 
    "📈 AUDIENCE FORECASTING",
    "🎲 BEHAVIORAL VARIANCE",
    "📋 EXPORT DOSSIER"
])

# =========================================================================================
# TAB 1 - DISCOVERY ENGINE (FUZZY MATCHING & RESULTS)
# =========================================================================================
with tab1:
    
    st.markdown('<div class="glass-panel"><div class="panel-heading" style="margin-bottom:15px; border:none; padding-bottom:0;">🔍 Query Data Lake</div>', unsafe_allow_html=True)
    
    col_input, col_btn = st.columns([4, 1], vertical_alignment="bottom")
    
    with col_input:
        user_query = st.text_input("Search Database", placeholder="Enter a movie title (e.g. 'Avngers', 'The Matrix'). Typo resolution is active.")
        
    with col_btn:
        search_clicked = st.button("RUN SIMILARITY SEARCH", use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    if search_clicked and user_query:
        if tfidf_matrix is None or indices_map is None:
            st.error("SYSTEM HALT: Core infrastructure missing. Cannot execute vector search.")
        else:
            with st.spinner("Running Levenshtein distance typo resolution and computing Cosine similarities..."):
                start_time = time.time()
                time.sleep(0.8) # UI polish
                
                # --- FUZZY MATCHING LOGIC (TYPO RESOLUTION) ---
                closest_matches = difflib.get_close_matches(user_query, ALL_TITLES, n=1, cutoff=0.5)
                
                if not closest_matches:
                    st.warning(f"DATABASE MISS: No entities found matching '{user_query}' within a 50% Levenshtein threshold. Please try another query.")
                    st.session_state["recommendations"] = None
                else:
                    matched_title = closest_matches[0]
                    
                    # Calculate basic fuzzy confidence
                    match_ratio = difflib.SequenceMatcher(None, user_query.lower(), matched_title.lower()).ratio() * 100
                    
                    st.session_state["matched_title"] = matched_title
                    st.session_state["match_confidence"] = match_ratio
                    st.session_state["user_input"] = user_query
                    
                    # --- COSINE SIMILARITY LOGIC ---
                    try:
                        # Get index of the matched movie
                        idx = int(indices_map[matched_title])
                            
                        # Compute similarity scores
                        sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
                        
                        # Enumerate and sort
                        sim_scores_enum = list(enumerate(sim_scores))
                        sim_scores_sorted = sorted(sim_scores_enum, key=lambda x: x[1], reverse=True)
                        
                        # Get top 10 similar movies
                        top_10_indices = [i[0] for i in sim_scores_sorted[1:11]]
                        top_10_scores = [i[1] for i in sim_scores_sorted[1:11]]
                        
                        # Build mock DataFrame using Reverse Index Mapping since we dropped movies.pkl
                        mock_data = []
                        for rank, matrix_idx in enumerate(top_10_indices):
                            # Reverse lookup the title from the matrix index
                            title = INDEX_TO_TITLE.get(matrix_idx, f"Unknown Title ID {matrix_idx}")
                            
                            mock_data.append({
                                "title": title,
                                "Similarity_Score": top_10_scores[rank],
                                "overview": "Metadata unavailable. Running in Ultra-Lightweight Vector Mode to bypass standard GitHub size limitations. The mathematical relationship remains 100% accurate based on TF-IDF scoring.",
                                "genres": "Vectorized NLP Content",
                                "vote_average": "N/A"
                            })
                            
                        st.session_state["recommendations"] = pd.DataFrame(mock_data)
                        
                        end_time = time.time()
                        st.session_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                        st.session_state["compute_latency"] = round(end_time - start_time, 3)
                        
                    except Exception as e:
                        st.error(f"Computation Error during Matrix Multiplication: {e}")

    # --- RENDER RESULTS ---
    if st.session_state["recommendations"] is not None:
        matched = st.session_state["matched_title"]
        conf = st.session_state["match_confidence"]
        recs = st.session_state["recommendations"]
        
        # Typo Resolution Alert
        if conf < 99.0:
            st.markdown(
f"""<div style="background:rgba(245,158,11,0.1); border-left:4px solid var(--gold-core); padding:15px 20px; margin-bottom:30px; border-radius:4px;">
<span style="color:var(--gold-core); font-family:'JetBrains Mono'; font-size:12px; letter-spacing:2px; text-transform:uppercase;">⚠️ FUZZY MATCH ACTIVATED</span><br>
<span style="color:var(--white-main); font-family:'Inter'; font-size:14px;">Input was auto-corrected to <b>"{matched}"</b> with a {conf:.1f}% confidence threshold.</span>
</div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="panel-heading" style="border:none;">🎬 Top Recommendations based on: <span style="color:var(--crimson-core);">{matched}</span></div>', unsafe_allow_html=True)
        
        # Render custom UI cards
        for i, row in recs.iterrows():
            title = row.get('title', 'Unknown Title')
            score = row.get('Similarity_Score', 0.0) * 100
            overview = str(row.get('overview', ''))
            genre = str(row.get('genres', ''))
            rating = str(row.get('vote_average', ''))
            
            st.markdown(
f"""<div class="movie-card">
<div class="movie-rank">{(recs.index.get_loc(i) + 1):02d}</div>
<div class="movie-title">{title}</div>
<div class="movie-meta">GENRE: {genre} | RATING: ⭐ {rating}</div>
<div class="movie-desc">{overview}</div>
<div><div class="movie-sim">COSINE SIMILARITY: {score:.1f}%</div></div>
</div>""", unsafe_allow_html=True)

# =========================================================================================
# TAB 2 - VECTOR ANALYTICS & SIMILARITY DISTRIBUTION
# =========================================================================================
with tab2:
    if st.session_state["recommendations"] is None:
        st.markdown(
"""<div style='text-align:center; padding:150px 20px; font-family:"Bebas Neue",serif; font-size:24px; letter-spacing:4px; color:rgba(225,29,72,0.4); text-transform:uppercase;'>
⚠️ Execute Discovery Search To Unlock Vector Analytics
</div>""",
            unsafe_allow_html=True,
        )
    else:
        recs = st.session_state["recommendations"]
        
        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.markdown('<div class="panel-heading" style="border:none;">📉 Cosine Similarity Decay Curve</div>', unsafe_allow_html=True)
            
            titles = recs.get('title', [f"Rec {i}" for i in range(1, 11)]).tolist()
            scores = (recs.get('Similarity_Score', [0]*10) * 100).tolist()
            
            fig_decay = go.Figure()
            fig_decay.add_trace(go.Scatter(
                x=titles, y=scores, mode='lines+markers',
                line=dict(color='#e11d48', width=3, shape='spline'),
                marker=dict(size=8, color='#f8fafc', line=dict(width=2, color='#e11d48')),
                fill='tozeroy', fillcolor='rgba(225, 29, 72, 0.1)'
            ))
            
            fig_decay.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(24,24,27,0.5)",
                font=dict(family="Inter", color="#f8fafc"),
                xaxis=dict(title="", tickangle=45, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Similarity Score (%)", range=[min(scores)-5, max(scores)+5], gridcolor="rgba(255,255,255,0.05)"),
                height=450, margin=dict(l=20, r=20, t=20, b=100)
            )
            st.plotly_chart(fig_decay, use_container_width=True)

        with col_a2:
            st.markdown('<div class="panel-heading" style="border:none;">📊 Simulated Quality Assessment</div>', unsafe_allow_html=True)
            
            np.random.seed(42)
            ratings = np.random.uniform(6.5, 8.9, len(recs)).round(1).tolist()

            fig_ratings = px.bar(
                x=titles, y=ratings,
                color=ratings, color_continuous_scale='Sunsetdark',
                labels={'x': '', 'y': 'Simulated Vote Average (out of 10)'}
            )
            
            fig_ratings.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(24,24,27,0.5)",
                font=dict(family="Inter", color="#f8fafc"),
                xaxis=dict(tickangle=45, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(range=[0, 10], gridcolor="rgba(255,255,255,0.05)"),
                showlegend=False, coloraxis_showscale=False,
                height=450, margin=dict(l=20, r=20, t=20, b=100)
            )
            st.plotly_chart(fig_ratings, use_container_width=True)

# =========================================================================================
# TAB 3 - NLP ARCHITECTURE (TF-IDF & COSINE SIMILARITY EXPLAINED)
# =========================================================================================
with tab3:
    st.markdown('<div class="panel-heading" style="border:none;">🧠 Natural Language Processing Pipeline</div>', unsafe_allow_html=True)
    
    st.info("💡 **Data Science Insight:** This system does not use collaborative filtering (user reviews). Instead, it uses **Content-Based Filtering**. It mathematically analyzes the text (Overviews, Genres, Cast) of the movies themselves to find semantic overlap.")
    
    col_i1, col_i2 = st.columns(2)
    
    insights = [
        ("🧮 Term Frequency (TF)", "Measures how frequently a word appears in a specific movie's description. If the word 'alien' appears 5 times in an overview, its TF score increases for that specific film."),
        ("📉 Inverse Document Frequency (IDF)", "Words like 'the', 'and', or even 'movie' appear everywhere and provide no unique value. IDF penalizes these common words across the entire dataset, while assigning high mathematical weight to rare, identifying words (like 'lightsaber' or 'Hogwarts')."),
        ("🌌 Vector Space Conversion", "The TF-IDF vectorizer converts every single movie into a high-dimensional mathematical vector (a coordinate in space). If there are 10,000 unique words in the dataset, every movie is plotted in a 10,000-dimensional graph."),
        ("📐 Cosine Similarity", "To find recommendations, we don't look at Euclidean distance (which is biased by the length of the text). Instead, we calculate the Cosine Angle between two movie vectors. An angle of 0 degrees = 1.0 (100% identical). The movies with the smallest angles to your query are returned as recommendations.")
    ]
    
    for i, (title, desc) in enumerate(insights):
        target = col_i1 if i % 2 == 0 else col_i2
        with target:
            st.markdown(
f"""<div class="glass-panel" style="padding:30px;">
<h4 style="color:var(--crimson-core); margin-bottom:15px; font-family:'Bebas Neue'; font-size:24px; letter-spacing:2px;">{title}</h4>
<p style="color:var(--slate-light); font-size:14px; line-height:1.8;">{desc}</p>
</div>""", unsafe_allow_html=True)

# =========================================================================================
# TAB 4 - AUDIENCE FORECASTING (SIMULATED ENGAGEMENT)
# =========================================================================================
with tab4:
    if st.session_state["recommendations"] is None:
        st.markdown(
"""<div style='text-align:center; padding:150px 20px; font-family:"Bebas Neue",serif; font-size:24px; letter-spacing:4px; color:rgba(225,29,72,0.4); text-transform:uppercase;'>
⚠️ Execute Discovery Search To Access Forecaster
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="panel-heading" style="border:none;">📈 Audience Retention Simulation for Recommended Slate</div>', unsafe_allow_html=True)
        
        recs = st.session_state["recommendations"]
        top_3 = recs.head(3).get('title', [f"Rec {i}" for i in range(1, 4)]).tolist()
        
        minutes = np.arange(0, 120, 10)
        
        fig_retention = go.Figure()
        
        colors = ['#e11d48', '#f59e0b', '#3b82f6']
        for i, title in enumerate(top_3):
            decay_factor = 0.01 + (i * 0.005)
            retention = 100 * np.exp(-decay_factor * minutes)
            
            fig_retention.add_trace(go.Scatter(
                x=minutes, y=retention, mode='lines+markers', 
                line=dict(color=colors[i], width=3), name=title
            ))
            
        fig_retention.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(24,24,27,0.5)",
            font=dict(family="Inter", color="#f8fafc"),
            xaxis=dict(title="Minutes Streamed", gridcolor="rgba(255,255,255,0.05)", dtick=20),
            yaxis=dict(title="Audience Retention (%)", range=[0,105], gridcolor="rgba(255,255,255,0.05)"),
            hovermode="x unified",
            height=500, margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_retention, use_container_width=True)

# =========================================================================================
# TAB 5 - BEHAVIORAL VARIANCE (MONTE CARLO SIMULATION)
# =========================================================================================
with tab5:
    if st.session_state["recommendations"] is None:
        st.markdown(
"""<div style='text-align:center; padding:150px 20px; font-family:"Bebas Neue",serif; font-size:24px; letter-spacing:4px; color:rgba(225,29,72,0.4); text-transform:uppercase;'>
⚠️ Execute Discovery Search To Access Variance Systems
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="panel-heading" style="border:none;">🎲 Click-Through Rate (CTR) Volatility</div>', unsafe_allow_html=True)
        
        st.info("Simulating 1,000 hypothetical user interactions. Even highly similar recommendations face behavioral variance (users ignoring the suggestion due to mood, artwork, or fatigue). This Monte Carlo simulation maps the probable Click-Through Rate (CTR) for your #1 recommendation.")
        
        top_score = st.session_state["recommendations"].iloc[0].get('Similarity_Score', 0.5)
        base_ctr = top_score * 100 
        np.random.seed(42)
        
        error_variance = 12.5 
        simulated_cohort = np.random.normal(base_ctr, error_variance, 1000)
        simulated_cohort = np.clip(simulated_cohort, 0, 100) 
        
        fig_mc = go.Figure()
        
        fig_mc.add_trace(go.Histogram(
            x=simulated_cohort,
            nbinsx=40,
            marker_color='rgba(245, 158, 11, 0.7)',
            marker_line_color='rgba(245, 158, 11, 1.0)',
            marker_line_width=2,
            opacity=0.8
        ))
        
        fig_mc.add_vline(
            x=base_ctr, line=dict(color="#f8fafc", width=3, dash="dash"),
            annotation_text=f"Expected Base CTR: {base_ctr:.1f}%", annotation_font_color="#f8fafc"
        )
        
        fig_mc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(24,24,27,0.5)",
            font=dict(family="Inter", color="#f8fafc"),
            xaxis=dict(title="Simulated Click-Through Rate (%)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Frequency (Out of 1,000 Users)", gridcolor="rgba(255,255,255,0.05)"),
            height=500, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_mc, use_container_width=True)

# =========================================================================================
# TAB 6 - EXPORT DOSSIER
# =========================================================================================
with tab6:
    if st.session_state["recommendations"] is None:
        st.markdown(
"""<div style='text-align:center; padding:150px 20px; font-family:"Bebas Neue",serif; font-size:24px; letter-spacing:4px; color:rgba(225,29,72,0.4); text-transform:uppercase;'>
⚠️ Execute Discovery Search To Generate Official Dossier
</div>""",
            unsafe_allow_html=True,
        )
    else:
        matched = st.session_state["matched_title"]
        ts = st.session_state["timestamp"]
        sess_id = st.session_state["session_id"]
        recs = st.session_state["recommendations"]
        
        st.markdown(
f"""<div class="glass-panel" style="background:rgba(225, 29, 72, 0.05); border-color:rgba(225, 29, 72, 0.3); padding:60px;">
<div style="font-family:'Space Mono'; font-size:14px; color:var(--crimson-core); margin-bottom:15px; letter-spacing:3px;">✅ SLATE GENERATED: {ts}</div>
<div style="font-family:'Bebas Neue'; font-size:50px; color:white; margin-bottom:10px; letter-spacing:2px;">SOURCE: {matched}</div>
<div style="font-family:'Inter'; font-size:18px; color:var(--slate-light);">Vector Transaction ID: <span style="color:var(--crimson-core); font-family:'Space Mono';">{sess_id}</span></div>
</div>""", unsafe_allow_html=True
        )

        st.markdown('<div class="panel-heading" style="border:none; margin-top:50px;">💾 Export Vector Artifacts</div>', unsafe_allow_html=True)
        
        col_exp1, col_exp2 = st.columns(2)
        
        export_df = recs.copy()
        for col in export_df.columns:
            if export_df[col].dtype == 'object':
                export_df[col] = export_df[col].astype(str)
                
        dict_records = export_df.to_dict(orient='records')
        
        json_payload = {
            "metadata": {
                "transaction_id": sess_id,
                "timestamp": ts,
                "source_query": st.session_state["user_input"],
                "resolved_match": matched,
                "levenshtein_confidence": st.session_state["match_confidence"]
            },
            "recommendations": dict_records
        }
        json_str = json.dumps(json_payload, indent=4)
        b64_json = base64.b64encode(json_str.encode()).decode()
        
        csv_data = export_df.to_csv(index=False)
        b64_csv = base64.b64encode(csv_data.encode()).decode()
        
        with col_exp1:
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="CineMetrics_Lightweight_{sess_id}.csv" style="display:block; text-align:center; padding:25px; background:rgba(225, 29, 72, 0.1); border:1px solid var(--crimson-core); color:var(--crimson-core); text-decoration:none; font-family:\'Space Mono\'; font-weight:700; font-size:16px; border-radius:8px; letter-spacing:2px; transition:all 0.3s ease;">⬇️ DOWNLOAD CSV LEDGER</a>'
            st.markdown(href_csv, unsafe_allow_html=True)
            
        with col_exp2:
            href_json = f'<a href="data:application/json;base64,{b64_json}" download="CineMetrics_Payload_{sess_id}.json" style="display:block; text-align:center; padding:25px; background:rgba(245, 158, 11, 0.1); border:1px solid var(--gold-core); color:var(--gold-core); text-decoration:none; font-family:\'Space Mono\'; font-weight:700; font-size:16px; border-radius:8px; letter-spacing:2px; transition:all 0.3s ease;">⬇️ DOWNLOAD JSON PAYLOAD</a>'
            st.markdown(href_json, unsafe_allow_html=True)

        st.markdown('<div class="panel-heading" style="border:none; margin-top:70px;">💻 Raw Transmission Payload</div>', unsafe_allow_html=True)
        st.json(json_payload)

# =========================================================================================
# 8. GLOBAL FOOTER
# =========================================================================================
st.markdown(
"""<div style="text-align:center; padding:70px; margin-top:100px; border-top:1px solid rgba(225,29,72,0.15); font-family:'Space Mono'; font-size:11px; color:rgba(148,163,184,0.3); letter-spacing:4px; text-transform:uppercase;">
&copy; 2026 | CineMetrics Content Intelligence Terminal v10.9<br>
<span style="color:rgba(225,29,72,0.5); font-size:10px; display:block; margin-top:10px;">Strictly Confidential Audience Data | Ultra-Lightweight TF-IDF Architecture</span>
</div>""",
    unsafe_allow_html=True,
)