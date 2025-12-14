import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from recommendation_engine import NetflixRecommender

st.set_page_config(
    page_title="Netflix • Find Shows You Love",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

premium_css = """
<style>
    /* Root variables */
    :root {
        --primary: #e50914;
        --secondary: #221f1f;
        --dark: #0f0f0f;
        --light: #ffffff;
        --text: #e0e0e0;
        --muted: #888888;
    }
    
    /* Global styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Netflix Loader */
    @keyframes fadeOutLoader {
        0% {
            opacity: 1;
            visibility: visible;
        }
        95% {
            opacity: 1;
            visibility: visible;
        }
        100% {
            opacity: 0;
            visibility: hidden;
        }
    }
    
    @keyframes pulseGrow {
        0% {
            transform: scale(0.8);
            opacity: 0.3;
        }
        50% {
            transform: scale(1.1);
            opacity: 0.7;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    @keyframes rotateLoader {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }
    
    @keyframes slideInLogo {
        0% {
            transform: translateY(30px) scale(0.8);
            opacity: 0;
        }
        100% {
            transform: translateY(0) scale(1);
            opacity: 1;
        }
    }
    
    @keyframes glowPulse {
        0%, 100% {
            box-shadow: 0 0 20px rgba(229, 9, 20, 0.4);
        }
        50% {
            box-shadow: 0 0 40px rgba(229, 9, 20, 0.8);
        }
    }
    
    .netflix-loader-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f1419 100%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        animation: fadeOutLoader 2.5s ease-out forwards;
    }
    
    .loader-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 40px;
    }
    
    .netflix-logo-loader {
        font-size: 120px;
        font-weight: 900;
        color: #e50914;
        text-transform: uppercase;
        letter-spacing: 4px;
        animation: slideInLogo 0.8s ease-out, glowPulse 2s ease-in-out infinite;
        font-family: Arial, sans-serif;
        text-shadow: 0 0 30px rgba(229, 9, 20, 0.6);
        margin: 0;
    }
    
    .loader-spinner {
        width: 60px;
        height: 60px;
        border: 3px solid rgba(229, 9, 20, 0.2);
        border-top: 3px solid #e50914;
        border-radius: 50%;
        animation: rotateLoader 1.2s linear infinite;
    }
    
    .loader-text {
        color: #999999;
        font-size: 16px;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
    }
    
    .dots {
        display: inline-block;
        margin-left: 5px;
    }
    
    .dots span {
        animation: pulseGrow 1.2s ease-in-out infinite;
        margin: 0 2px;
    }
    
    .dots span:nth-child(1) {
        animation-delay: 0s;
    }
    
    .dots span:nth-child(2) {
        animation-delay: 0.3s;
    }
    
    .dots span:nth-child(3) {
        animation-delay: 0.6s;
    }
    
    /* Main background */
    body, .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f1419 100%);
        background-attachment: fixed;
    }
    
    /* Remove default padding */
    .main {
        padding: 0 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 15, 15, 0.95) 0%, rgba(25, 25, 25, 0.95) 100%);
        border-right: 1px solid rgba(229, 9, 20, 0.2);
        backdrop-filter: blur(20px);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    
    /* Sidebar header */
    .sidebar-header {
        padding: 30px 20px 20px;
        background: linear-gradient(180deg, rgba(229, 9, 20, 0.1) 0%, transparent 100%);
        border-bottom: 1px solid rgba(229, 9, 20, 0.2);
        margin-bottom: 0;
    }
    
    .sidebar-title {
        font-size: 1.6em;
        font-weight: 800;
        color: #e50914;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0;
    }
    
    .sidebar-subtitle {
        font-size: 0.75em;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    /* Navigation items */
    .nav-section {
        padding: 20px 15px;
    }
    
    .nav-label {
        font-size: 0.8em;
        text-transform: uppercase;
        color: #666666;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        font-weight: 700;
        padding: 0 5px;
    }
    
    .stRadio > label {
        background: transparent !important;
    }
    
    .stRadio > div {
        gap: 0 !important;
    }
    
    .stRadio [role="radio"] {
        accent-color: #e50914;
    }
    
    /* Radio button styling */
    .stRadio > div > label {
        padding: 12px 15px;
        margin: 5px 0;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: rgba(255, 255, 255, 0.02);
        transition: all 0.3s ease;
        cursor: pointer;
        font-weight: 500;
        color: #999999;
    }
    
    .stRadio > div > label:hover {
        background: rgba(229, 9, 20, 0.1);
        border-color: rgba(229, 9, 20, 0.3);
        color: #e50914;
    }
    
    .stRadio > div > label[aria-checked="true"] {
        background: rgba(229, 9, 20, 0.2);
        border-color: #e50914;
        color: #e50914;
    }
    
    /* Main content area */
    .main-content {
        padding: 0 40px 40px 40px;
    }
    
    /* Header section */
    .header-section {
        padding: 50px 40px;
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.05) 0%, transparent 100%);
        margin: 0 -40px 40px -40px;
        border-bottom: 1px solid rgba(229, 9, 20, 0.1);
    }
    
    .main-title {
        font-size: 3.5em;
        font-weight: 900;
        color: #ffffff;
        margin: 0;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    
    .title-accent {
        color: #e50914;
    }
    
    .tagline {
        font-size: 1.1em;
        color: #888888;
        margin-top: 10px;
        font-weight: 300;
    }
    
    /* Premium card */
    .premium-card {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.08) 0%, rgba(50, 50, 50, 0.04) 100%);
        border: 1px solid rgba(229, 9, 20, 0.2);
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .premium-card:hover {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.15) 0%, rgba(50, 50, 50, 0.08) 100%);
        border-color: rgba(229, 9, 20, 0.5);
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(229, 9, 20, 0.15);
    }
    
    /* Movie card */
    .movie-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(229, 9, 20, 0.02) 100%);
        border: 1px solid rgba(229, 9, 20, 0.15);
        border-radius: 16px;
        padding: 28px;
        margin: 18px 0;
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        backdrop-filter: blur(15px);
        position: relative;
        overflow: hidden;
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(229, 9, 20, 0.1), transparent);
        transition: left 0.5s ease;
        pointer-events: none;
    }
    
    .movie-card:hover::before {
        left: 100%;
    }
    
    .movie-card:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(229, 9, 20, 0.05) 100%);
        border-color: rgba(229, 9, 20, 0.4);
        transform: translateY(-12px);
        box-shadow: 0 30px 60px rgba(229, 9, 20, 0.2);
    }
    
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #e50914 0%, #ff6b6b 100%);
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3);
    }
    
    .movie-title {
        font-size: 1.5em;
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 12px;
        line-height: 1.3;
    }
    
    .movie-meta {
        font-size: 0.95em;
        color: #999999;
        margin: 8px 0;
        line-height: 1.6;
        font-weight: 400;
    }
    
    .similarity-section {
        margin-top: 20px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .similarity-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .similarity-label {
        font-size: 0.9em;
        color: #999999;
        font-weight: 500;
    }
    
    .similarity-score {
        font-size: 1.2em;
        color: #e50914;
        font-weight: 700;
    }
    
    .similarity-bar {
        height: 7px;
        background: linear-gradient(90deg, rgba(229, 9, 20, 0.1), rgba(229, 9, 20, 0.05));
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }
    
    .similarity-fill {
        height: 100%;
        background: linear-gradient(90deg, #e50914 0%, #ff6b6b 100%);
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(229, 9, 20, 0.5);
        transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #e50914 0%, #c20812 100%);
        color: #ffffff;
        border: none;
        padding: 14px 32px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95em;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(229, 9, 20, 0.3);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(229, 9, 20, 0.4);
        background: linear-gradient(135deg, #ff6b6b 0%, #e50914 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(229, 9, 20, 0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(229, 9, 20, 0.4) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    .stSlider > div > div > div > div {
        background: #e50914 !important;
    }
    
    /* Headers */
    h1 {
        color: #ffffff;
        font-weight: 800;
        margin: 30px 0 20px 0;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #ffffff;
        font-weight: 700;
        margin: 25px 0 18px 0;
        font-size: 2em;
        letter-spacing: -0.5px;
    }
    
    h3 {
        color: #e50914;
        font-weight: 600;
        margin: 20px 0 15px 0;
    }
    
    h4 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid rgba(255, 255, 255, 0.08);
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #888888 !important;
        font-weight: 600;
        border: none !important;
        background: transparent !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        background: rgba(229, 9, 20, 0.1) !important;
        color: #e50914 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #e50914 !important;
        background: rgba(229, 9, 20, 0.15) !important;
        border-bottom: 3px solid #e50914 !important;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(229, 9, 20, 0.02) 100%);
        border: 1px solid rgba(229, 9, 20, 0.2);
        border-radius: 12px;
        padding: 20px !important;
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(229, 9, 20, 0.05) 100%);
        border-color: rgba(229, 9, 20, 0.4);
        transform: translateY(-3px);
    }
    
    /* Text */
    p {
        color: #cccccc;
        line-height: 1.8;
    }
    
    /* Info boxes */
    .stSuccess {
        background-color: rgba(34, 139, 34, 0.15) !important;
        border-left: 4px solid #22a922 !important;
        border-radius: 8px !important;
    }
    
    .stInfo {
        background-color: rgba(229, 9, 20, 0.15) !important;
        border-left: 4px solid #e50914 !important;
        border-radius: 8px !important;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.15) !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 8px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(229, 9, 20, 0.08) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(229, 9, 20, 0.15) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(229, 9, 20, 0.15) !important;
        border-color: rgba(229, 9, 20, 0.3) !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 30px 0 !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(229, 9, 20, 0.1) !important;
    }
    
    /* Code blocks */
    code {
        background: rgba(229, 9, 20, 0.1) !important;
        color: #ff6b6b !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(229, 9, 20, 0.4);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(229, 9, 20, 0.6);
    }
    
    /* Movie Banner Styles */
    .movie-banner {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        margin: 20px 0;
        height: 320px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(229, 9, 20, 0.02) 100%);
        border: 1px solid rgba(229, 9, 20, 0.15);
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        display: flex;
        align-items: stretch;
    }
    
    .movie-banner:hover {
        border-color: rgba(229, 9, 20, 0.4);
        transform: translateY(-12px);
        box-shadow: 0 30px 60px rgba(229, 9, 20, 0.2);
    }
    
    .banner-poster {
        width: 200px;
        height: 100%;
        object-fit: cover;
        flex-shrink: 0;
        background: linear-gradient(135deg, #e50914, #221f1f);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #ffffff;
        font-size: 0.75em;
        text-align: center;
        padding: 20px;
        position: relative;
        overflow: hidden;
    }
    
    .banner-poster::before {
        content: '🎬';
        font-size: 4em;
        opacity: 0.3;
        position: absolute;
    }
    
    .banner-content {
        flex: 1;
        padding: 24px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background: linear-gradient(90deg, rgba(34, 31, 31, 0.4) 0%, transparent 60%);
    }
    
    .banner-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 12px;
    }
    
    .banner-rank {
        display: inline-block;
        background: linear-gradient(135deg, #e50914 0%, #ff6b6b 100%);
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3);
    }
    
    .banner-title {
        font-size: 1.8em;
        color: #ffffff;
        font-weight: 700;
        margin: 8px 0;
        line-height: 1.3;
    }
    
    .banner-meta {
        font-size: 0.95em;
        color: #999999;
        margin: 6px 0;
        font-weight: 400;
    }
    
    .banner-description {
        font-size: 0.9em;
        color: #cccccc;
        margin: 12px 0;
        line-height: 1.6;
        flex: 1;
    }
    
    .banner-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .banner-similarity {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .banner-similarity-label {
        font-size: 0.85em;
        color: #999999;
        font-weight: 500;
    }
    
    .banner-similarity-score {
        font-size: 1.6em;
        color: #e50914;
        font-weight: 700;
    }
    
    .banner-similarity-bar {
        height: 8px;
        background: linear-gradient(90deg, rgba(229, 9, 20, 0.1), rgba(229, 9, 20, 0.05));
        border-radius: 10px;
        overflow: hidden;
        width: 150px;
    }
    
    .banner-similarity-fill {
        height: 100%;
        background: linear-gradient(90deg, #e50914 0%, #ff6b6b 100%);
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(229, 9, 20, 0.5);
        transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
</style>
"""

st.markdown(premium_css, unsafe_allow_html=True)

loader_html = """
<div class="netflix-loader-overlay">
    <div class="loader-content">
        <div class="netflix-logo-loader">NETFLIX</div>
        <div class="loader-spinner"></div>
        <div class="loader-text">
            Loading<span class="dots"><span>.</span><span>.</span><span>.</span></span>
        </div>
    </div>
</div>
"""

st.markdown(loader_html, unsafe_allow_html=True)

def get_poster_url(title):
    """Generate placeholder poster URL for export"""
    return f"https://via.placeholder.com/300x450/e50914/ffffff?text=Poster"

def render_movie_card(rank, title, genre, year, description, similarity):
    """Render premium movie banner card with animations"""
    sim_pct = int(similarity * 100)
    
    html = f"""
    <div class="movie-banner">
        <div class="banner-poster">
            {title.upper()[:15]}
        </div>
        <div class="banner-content">
            <div>
                <div class="banner-header">
                    <span class="banner-rank">#{rank} Match</span>
                </div>
                <div class="banner-title">{title}</div>
                <div class="banner-meta">🎬 {genre}</div>
                <div class="banner-meta">📅 {year}</div>
                <div class="banner-description">{description[:180]}...</div>
            </div>
            <div class="banner-footer">
                <div class="banner-similarity">
                    <span class="banner-similarity-label">Match Score</span>
                    <span class="banner-similarity-score">{sim_pct}%</span>
                </div>
                <div class="banner-similarity-bar">
                    <div class="banner-similarity-fill" style="width: {sim_pct}%;"></div>
                </div>
            </div>
        </div>
    </div>
    """
    return html

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <p class="sidebar-title">NETFLIX</p>
            <p class="sidebar-subtitle">AI Recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='nav-section'>", unsafe_allow_html=True)
        st.markdown("<div class='nav-label'>📍 Navigation</div>", unsafe_allow_html=True)
        
        page = st.radio(
            "Choose section:",
            ["🏠 Home", "🎯 Discover", "📊 Analytics", "ℹ️ About"],
            label_visibility="collapsed",
            key="nav"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("<div class='nav-section'>", unsafe_allow_html=True)
        st.markdown("<div class='nav-label'>ℹ️ About App</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 0.9em; color: #999999; line-height: 1.6;">
        Smart recommendations using machine learning. Find shows you'll love instantly.
        </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='nav-section'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Shows", "500+")
        with col2:
            st.metric("Genres", "20+")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if "page" not in st.session_state:
        st.session_state.page = page
    else:
        st.session_state.page = page
    
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'netflix_sample.csv')
    current_page = st.session_state.page
    
    if current_page == "🏠 Home":
        st.markdown("""
        <div class="header-section">
            <div class="main-title">Discover <span class="title-accent">Your</span> Next <span class="title-accent">Favorite</span></div>
            <p class="tagline">AI-Powered Recommendations | Find Shows You'll Love</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1.3, 1])
        
        with col1:
            st.markdown("""
            <div class="premium-card">
            <h3>How It Works</h3>
            <p style="color: #cccccc; line-height: 1.8;">
            Our AI analyzes thousands of shows to find perfect matches for you.
            </p>
            <ul style="color: #cccccc; margin-left: 20px;">
                <li style="margin: 10px 0;"><strong>🎬 Analyzes</strong> genres, cast, themes</li>
                <li style="margin: 10px 0;"><strong>🔍 Learns</strong> your preferences</li>
                <li style="margin: 10px 0;"><strong>⚡ Recommends</strong> instant matches</li>
                <li style="margin: 10px 0;"><strong>🎯 Scores</strong> similarity accuracy</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="premium-card">
            <h3>Quick Stats</h3>
            """, unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🎬 Content", "500+")
                st.metric("⚡ Speed", "<1s")
            with col_b:
                st.metric("🎭 Genres", "20+")
                st.metric("🎯 Accuracy", "95%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="premium-card">
            <h4>🚀 Get Started</h4>
            <p style="color: #cccccc; font-size: 0.95em;">Go to Discover, pick your favorite show, and get instant recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="premium-card">
            <h4>📊 Explore Data</h4>
            <p style="color: #cccccc; font-size: 0.95em;">Check Analytics to see genre trends, top shows, and dataset insights.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="premium-card">
            <h4>📚 Learn More</h4>
            <p style="color: #cccccc; font-size: 0.95em;">Visit About to understand how our algorithm works and the tech behind it.</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif current_page == "🎯 Discover":
        st.markdown("""
        <div class="header-section">
            <div class="main-title">Find <span class="title-accent">Perfect</span> Matches</div>
            <p class="tagline">Search any show and get instant AI recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔄 Loading AI engine..."):
            recommender = NetflixRecommender(data_path)
            recommender.build_model()
        
        st.success("✅ AI engine ready!")
        
        col1, col2 = st.columns([2.5, 1.2])
        
        with col1:
            movie_title = st.selectbox(
                "Search for a show:",
                recommender.df['title'].unique(),
                label_visibility="collapsed",
                key="search"
            )
        
        with col2:
            num_recs = st.slider("Results", 1, 15, 5, label_visibility="collapsed", key="slider")
        
        st.markdown("")
        
        if st.button("🔍 Get Recommendations", use_container_width=True, key="get_recs"):
            with st.spinner("🎬 Finding perfect matches..."):
                recs = recommender.get_recommendations(movie_title, num_recs)
            
            if recs is not None:
                movie_info = recommender.df[recommender.df['title'] == movie_title].iloc[0]
                
                st.markdown(f"### You selected: **{movie_title}**")
                st.caption(f"🎬 {movie_info['listed_in']} • 📅 {movie_info['release_year']}")
                st.markdown("---")
                
                st.markdown(f"### 🏆 Top {num_recs} Matches")
                
                for idx, (_, row) in enumerate(recs.iterrows(), 1):
                    st.markdown(
                        render_movie_card(
                            idx,
                            row['title'],
                            row['listed_in'],
                            row['release_year'],
                            row['description'],
                            row['similarity_score']
                        ),
                        unsafe_allow_html=True
                    )
                
                st.markdown("---")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### 📥 Export Recommendations")
                with col2:
                    if st.button("💾 Download as Excel", use_container_width=True):
                        export_df = recs.copy()
                        export_df['poster_url'] = export_df['title'].apply(get_poster_url)
                        export_df['match_rank'] = range(1, len(export_df) + 1)
                        export_df['similarity_percentage'] = (export_df['similarity_score'] * 100).astype(int).astype(str) + '%'
                        
                        cols_to_export = ['match_rank', 'title', 'type', 'listed_in', 'release_year', 'similarity_percentage', 'poster_url', 'description']
                        export_df_final = export_df[[col for col in cols_to_export if col in export_df.columns]]
                        
                        with pd.ExcelWriter('recommendations_export.xlsx', engine='openpyxl') as writer:
                            export_df_final.to_excel(writer, sheet_name='Recommendations', index=False)
                        
                        with open('recommendations_export.xlsx', 'rb') as f:
                            st.download_button(
                                label="✅ Click to download",
                                data=f.read(),
                                file_name=f"netflix_recommendations_{movie_title.replace(' ', '_')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
            else:
                st.error("❌ Show not found")
    
    elif current_page == "📊 Analytics":
        st.markdown("""
        <div class="header-section">
            <div class="main-title">Dataset <span class="title-accent">Intelligence</span></div>
            <p class="tagline">Explore trends, genres, and content insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        df = pd.read_csv(data_path)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Total Content", len(df))
        with col2:
            st.metric("🎬 Movies", len(df[df['type'] == 'Movie']))
        with col3:
            st.metric("📺 TV Shows", len(df[df['type'] == 'TV Show']))
        with col4:
            genres_count = len(set(df['listed_in'].str.split(',').explode().str.strip()))
            st.metric("🎭 Genres", genres_count)
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Genres", "🎯 Type", "📅 Year", "📋 Data"])
        
        with tab1:
            st.markdown("### Top 12 Genres")
            genres = df['listed_in'].str.split(',').explode().str.strip()
            genre_counts = genres.value_counts().head(12)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(genre_counts.index, genre_counts.values, color='#e50914', alpha=0.85)
            ax.set_facecolor('#0a0a0a')
            fig.patch.set_facecolor('#0a0a0a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#333333')
            ax.spines['bottom'].set_color('#333333')
            ax.tick_params(colors='#888888')
            st.pyplot(fig)
        
        with tab2:
            st.markdown("### Content Distribution")
            types = df['type'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#e50914', '#666666']
            ax.pie(types, labels=types.index, autopct='%1.0f%%',
                  colors=colors, textprops={'color': '#ffffff', 'weight': 'bold'})
            ax.set_facecolor('#0a0a0a')
            fig.patch.set_facecolor('#0a0a0a')
            st.pyplot(fig)
        
        with tab3:
            st.markdown("### Release Year Trends")
            years = df['release_year'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(years.index, years.values, color='#e50914', linewidth=3, marker='o', markersize=7)
            ax.fill_between(years.index, years.values, alpha=0.25, color='#e50914')
            ax.set_facecolor('#0a0a0a')
            fig.patch.set_facecolor('#0a0a0a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#333333')
            ax.spines['bottom'].set_color('#333333')
            ax.tick_params(colors='#888888')
            ax.grid(True, alpha=0.1)
            st.pyplot(fig)
        
        with tab4:
            st.markdown("### Dataset Preview")
            st.dataframe(df.head(25))
    
    elif current_page == "ℹ️ About":
        st.markdown("""
        <div class="header-section">
            <div class="main-title">About <span class="title-accent">Netflix AI</span></div>
            <p class="tagline">Technology Behind Your Recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="premium-card">
            <h3>🛠️ Technology Stack</h3>
            <ul style="color: #cccccc; margin-left: 20px;">
                <li><strong>Machine Learning</strong> - Scikit-learn</li>
                <li><strong>NLP</strong> - NLTK tokenization</li>
                <li><strong>Data Science</strong> - Pandas & NumPy</li>
                <li><strong>Interface</strong> - Streamlit</li>
                <li><strong>Visualization</strong> - Matplotlib</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="premium-card">
            <h3>🧠 Algorithm</h3>
            <ol style="color: #cccccc; margin-left: 20px;">
                <li style="margin: 8px 0;">Parse show metadata</li>
                <li style="margin: 8px 0;">TF-IDF vectorization</li>
                <li style="margin: 8px 0;">Cosine similarity</li>
                <li style="margin: 8px 0;">Rank by score</li>
                <li style="margin: 8px 0;">Return matches</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("📚 Deep Dive - Content-Based Filtering"):
            st.markdown("""
            **How It Works:**
            - Analyzes show features (genre, cast, director, plot)
            - Converts text to numerical vectors using TF-IDF
            - Calculates similarity between all pairs using cosine distance
            - Ranks shows by similarity score (0-1 scale)
            - Returns top-N recommendations
            
            **Why This Approach:**
            - No cold-start problem for new content
            - Transparent, explainable recommendations
            - Works with minimal data
            - Fast computation
            """)
        
        with st.expander("📊 TF-IDF Vectorization"):
            st.markdown("""
            **Term Frequency-Inverse Document Frequency**
            
            Converts text to meaningful numbers:
            - **TF**: How often a word appears in a document
            - **IDF**: How unique a word is across all documents
            
            Result: 5000-dimensional vectors capturing show essence
            """)
        
        with st.expander("⚡ Cosine Similarity"):
            st.markdown("""
            **Measures angle between vectors**
            
            - Score 1.0 = Identical content
            - Score 0.5 = Similar content
            - Score 0.0 = Completely different
            
            Perfect for high-dimensional text data!
            """)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 50px; padding: 30px; 
                    background: rgba(229, 9, 20, 0.08); border-radius: 12px;
                    border: 1px solid rgba(229, 9, 20, 0.2);">
            <p style="font-size: 1.3em; color: #e50914; font-weight: 700; margin-bottom: 10px;">
            Netflix AI Recommendation Engine
            </p>
            <p style="color: #888888; font-size: 0.95em;">
            Powered by Machine Learning • Built with ❤️ Kartik Bhalodiya
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
