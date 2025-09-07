import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import các hàm tiện ích
from data_utils import load_data, add_technical_indicators
from model_utils import preprocess_data, build_model, train_model, evaluate_model

# Import các thư viện ML cần thiết cho app.py
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from predict_future import predict_future
from arima_model import (
    prepare_data_for_arima, train_arima_model, predict_arima, 
    evaluate_arima_model, compare_models_performance, check_stationarity
)

# Cấu hình trang
st.set_page_config(
    page_title="VNIndex Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS tùy chỉnh - Thiết kế chuyên nghiệp theo phong cách Investing.com
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Reset và base styling */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #f8fafc;
    }
    
    /* Header chính - Investing.com style */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a365d;
        text-align: center;
        margin-bottom: 2.5rem;
        background: linear-gradient(135deg, #2d3748 0%, #1a365d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    /* Navigation và Tabs - Professional top navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        border-radius: 16px;
        padding: 6px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin-bottom: 32px;
        border: 1px solid #e2e8f0;
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        -webkit-overflow-scrolling: touch;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 56px;
        padding: 0 28px;
        border-radius: 12px;
        background-color: transparent;
        color: #4a5568;
        font-weight: 500;
        font-size: 15px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        margin: 2px;
        position: relative;
        overflow: hidden;
        flex-shrink: 0;
        min-width: fit-content;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(45, 55, 72, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        left: 100%;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #f7fafc 100%);
        color: #2d3748;
        transform: translateY(-1px);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: white;
        box-shadow: 0 4px 16px rgba(45, 55, 72, 0.3);
        transform: translateY(-1px);
    }
    
    /* Cards và Widgets - Investing.com style */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #38a169 0%, #48bb78 50%, #68d391 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.12);
        border-color: #cbd5e0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 8px;
        line-height: 1.1;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    
    .metric-change-positive {
        color: #38a169;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .metric-change-negative {
        color: #e53e3e;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .metric-sublabel {
        font-size: 0.75rem;
        color: #a0aec0;
        font-weight: 400;
        margin-top: 4px;
    }
    
    /* Sidebar styling - Dark professional theme */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
        color: white;
        border-right: 1px solid #4a5568;
    }
    
    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stFileUploader label,
    .css-1d391kg .stMarkdown {
        color: #e2e8f0;
        font-weight: 500;
    }
    
    .css-1d391kg .stSelectbox > div > div {
        background-color: #2d3748;
        border-color: #4a5568;
        color: white;
    }
    
    /* Buttons - Professional financial platform style */
    .stButton > button {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 16px 32px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(45, 55, 72, 0.25);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(45, 55, 72, 0.35);
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary buttons - Investing.com blue */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #3182ce 0%, #2c5aa0 100%);
        box-shadow: 0 6px 20px rgba(49, 130, 206, 0.25);
    }
    
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2c5aa0 0%, #3182ce 100%);
        box-shadow: 0 8px 32px rgba(49, 130, 206, 0.35);
    }
    
    /* Success buttons */
    .success-btn {
        background: linear-gradient(135deg, #38a169 0%, #48bb78 100%) !important;
        box-shadow: 0 6px 20px rgba(56, 161, 105, 0.25) !important;
    }
    
    /* Form elements - Clean professional styling */
    .stSelectbox > div > div,
    .stSlider > div > div,
    .stNumberInput > div > div,
    .stDateInput > div > div,
    .stMultiSelect > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background-color: #ffffff;
    }
    
    .stSelectbox > div > div:focus-within,
    .stSlider > div > div:focus-within,
    .stNumberInput > div > div:focus-within,
    .stDateInput > div > div:focus-within,
    .stMultiSelect > div > div:focus-within {
        border-color: #3182ce;
        box-shadow: 0 0 0 4px rgba(49, 130, 206, 0.1);
        transform: translateY(-1px);
    }
    
    /* Alerts và Messages - Enhanced visual hierarchy */
    .stAlert {
        border-radius: 16px;
        border-left-width: 6px;
        font-weight: 500;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }
    
    .stAlert[data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        border-left-color: #3182ce;
        color: #2c5aa0;
    }
    
    .stAlert[data-baseweb="notification"][kind="success"] {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left-color: #38a169;
        color: #276749;
    }
    
    .stAlert[data-baseweb="notification"][kind="warning"] {
        background: linear-gradient(135deg, #fffaf0 0%, #fbd38d 100%);
        border-left-color: #d69e2e;
        color: #b7791f;
    }
    
    .stAlert[data-baseweb="notification"][kind="error"] {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left-color: #e53e3e;
        color: #c53030;
    }
    
    /* Section headers - Professional typography */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 24px;
        padding-bottom: 12px;
        border-bottom: 3px solid #e2e8f0;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #3182ce 0%, #63b3ed 100%);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3182ce 0%, #63b3ed 100%);
        border-radius: 12px;
        height: 12px;
    }
    
    /* DataFrames - Clean table styling */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Plotly charts container */
    .js-plotly-plot {
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
        overflow: hidden;
        border: 1px solid #e2e8f0;
        margin: 0 auto;
        max-width: 100%;
    }
    
    /* Plotly chart responsive adjustments */
    .stPlotlyChart {
        width: 100% !important;
        margin: 0 auto;
        padding: 0 10px;
    }
    
    .stPlotlyChart > div {
        width: 100% !important;
        margin: 0 auto;
    }
    
    /* Metrics styling - Investing.com inspired */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #cbd5e0;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    }
    
    .stFileUploader:hover {
        border-color: #3182ce;
        background: linear-gradient(135deg, #ebf8ff 0%, #f8fafc 100%);
    }
    
    /* Footer */
    .footer {
        margin-top: 64px;
        padding: 32px 0;
        border-top: 2px solid #e2e8f0;
        text-align: center;
        color: #718096;
        font-size: 14px;
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        font-weight: 500;
    }
    
    /* Loading states */
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: calc(200px + 100%) 0; }
    }
    
    .loading {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200px 100%;
        animation: shimmer 1.5s infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
        }
        
        .metric-card {
            padding: 20px;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap;
            gap: 4px;
            padding: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0 12px;
            font-size: 12px;
            height: 44px;
            min-width: auto;
            flex: 1;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
    
    /* Extra small screens (phones) */
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.8rem;
            text-align: center;
            margin-bottom: 16px;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            flex-direction: column;
            gap: 6px;
            padding: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            width: 100%;
            padding: 12px 16px;
            font-size: 14px;
            height: 48px;
            margin: 2px 0;
            text-align: center;
            border-radius: 8px;
        }
        
        .metric-card {
            padding: 16px;
            margin: 12px 0;
        }
        
        .metric-value {
            font-size: 1.8rem;
        }
        
        .metric-label {
            font-size: 0.8rem;
        }
    }
    
    /* Custom scrollbar - Professional styling */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #cbd5e0 0%, #a0aec0 100%);
        border-radius: 6px;
        border: 2px solid #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #a0aec0 0%, #718096 100%);
    }
    
    /* Enhanced visual hierarchy */
    h1, h2, h3 {
        color: #2d3748;
        font-weight: 700;
        line-height: 1.2;
    }
    
    /* Market data styling - Financial platform look */
    .market-data-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 20px;
        padding: 32px;
        color: white;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.2);
        margin: 24px 0;
    }
    
    .market-data-item {
        border-bottom: 1px solid #4a5568;
        padding: 16px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .market-data-item:last-child {
        border-bottom: none;
    }
    
    /* Ẩn các nút Streamlit không cần thiết */
    .stDeployButton {
        display: none;
    }
    
    button[data-testid="stToolbarActionButton"] {
        display: none;
    }
    
    div[data-testid="stToolbar"] {
        display: none;
    }
    
    /* Ẩn menu hamburger và các nút góc phải */
    .stAppHeader {
        display: none;
    }
    
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Ẩn footer Streamlit */
    footer {
        display: none;
    }
    
    .streamlit-footer {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề chính - Professional header
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 class="main-header">VNIndex Stock Prediction Dashboard</h1>
    <p style="font-size: 1.2rem; color: #718096; font-weight: 500; margin-top: -1rem;">
        Nền tảng phân tích và dự đoán chứng khoán bằng AI
    </p>
    <div style="width: 100px; height: 3px; background: linear-gradient(90deg, #3182ce 0%, #63b3ed 100%); margin: 1rem auto; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'y_test_inv' not in st.session_state:
    st.session_state.y_test_inv = None
if 'y_pred_inv' not in st.session_state:
    st.session_state.y_pred_inv = None

# Sidebar - Professional data loading section
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h2 class="st-al" style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">📁 Quản lý dữ liệu</h2>
    <div style="width: 60px; height: 2px; background: linear-gradient(90deg, #38a169 0%, #48bb78 100%); margin: 0 auto; border-radius: 1px;"></div>
</div>
""", unsafe_allow_html=True)

# Upload file hoặc sử dụng file mặc định
uploaded_file = st.sidebar.file_uploader(
    "📤 Tải lên file CSV", 
    type=['csv'],
    help="Chọn file CSV chứa dữ liệu chứng khoán"
)

if uploaded_file is not None:
    st.session_state.data = load_data(uploaded_file)
else:
    # Sử dụng file mặc định (đường dẫn tương đối)
    default_file = os.path.join(os.path.dirname(__file__), "data/VNI_2020_2025_FINAL.csv")
    if os.path.exists(default_file):
        st.session_state.data = load_data(default_file)
        st.sidebar.success("✅ Đã tải file dữ liệu mặc định")
    else:
        st.sidebar.warning("⚠️ Vui lòng tải lên file dữ liệu CSV")

# Kiểm tra dữ liệu đã được tải
if st.session_state.data is not None:
    # Thêm chỉ báo kỹ thuật
    st.session_state.data = add_technical_indicators(st.session_state.data)
    
    # Tạo các tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dữ liệu lịch sử", 
        "⚙️ Cấu hình mô hình", 
        "🧪 Huấn luyện", 
        "📈 Dự đoán",
        "🔄 So sánh ARIMA",
        "💾 Quản lý mô hình"
    ])
    
    with tab1:
        st.markdown('<h2 class="section-header">📊 Dữ liệu lịch sử và Chỉ báo kỹ thuật</h2>', unsafe_allow_html=True)
        
        # Hiển thị thông tin cơ bản trong cards chuyên nghiệp
        st.markdown('<h3 style="color: #4a5568; font-weight: 600; margin: 2rem 0 1rem 0;">💹 Tổng quan thị trường hôm nay</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tổng số phiên giao dịch</div>
                <div class="metric-value">{len(st.session_state.data):,}</div>
                <div class="metric-sublabel">Dữ liệu khả dụng</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            latest_price = st.session_state.data['Close'].iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Giá đóng cửa</div>
                <div class="metric-value">{latest_price:,.0f}</div>
                <div class="metric-sublabel">VNĐ</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            price_change = st.session_state.data['Close'].iloc[-1] - st.session_state.data['Close'].iloc[-2]
            change_class = "metric-change-positive" if price_change >= 0 else "metric-change-negative"
            change_icon = "▲" if price_change >= 0 else "▼"
            change_text = "Tăng" if price_change >= 0 else "Giảm"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Thay đổi phiên</div>
                <div class="metric-value {change_class}">{change_icon} {abs(price_change):,.0f}</div>
                <div class="metric-sublabel">{change_text} so với hôm qua</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            volatility = st.session_state.data['Close'].pct_change().std() * 100
            volatility_color = "#e53e3e" if volatility > 2 else "#38a169" if volatility < 1 else "#d69e2e"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Độ biến động</div>
                <div class="metric-value" style="color: {volatility_color};">{volatility:.2f}%</div>
                <div class="metric-sublabel">Độ lệch chuẩn</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Khoảng cách
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Chọn khoảng thời gian hiển thị
        st.markdown('<h3 class="section-header">📅 Cấu hình khoảng thời gian phân tích</h3>', unsafe_allow_html=True)
        
        # Layout 3 cột cho date selection
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            start_date = st.date_input(
                "📅 Từ ngày",
                value=st.session_state.data.index[-365] if len(st.session_state.data) > 365 else st.session_state.data.index[0],
                help="Chọn ngày bắt đầu phân tích"
            )
        
        with col2:
            end_date = st.date_input(
                "📅 Đến ngày", 
                value=st.session_state.data.index[-1],
                help="Chọn ngày kết thúc phân tích"
            )
            
        with col3:
            # Quick selection
            st.markdown("**⚡ Chọn nhanh khoảng thời gian:**")
            quick_periods = {
                "1 tháng": 30,
                "3 tháng": 90, 
                "6 tháng": 180,
                "1 năm": 365,
                "Tất cả": len(st.session_state.data)
            }
            
            selected_period = st.selectbox(
                "Khoảng thời gian",
                list(quick_periods.keys()),
                index=3,
                help="Chọn khoảng thời gian phổ biến"
            )
            
            if selected_period and selected_period != "Tất cả":
                days_back = quick_periods[selected_period]
                start_date = st.session_state.data.index[-min(days_back, len(st.session_state.data))]
            elif selected_period == "Tất cả":
                start_date = st.session_state.data.index[0]
                end_date = st.session_state.data.index[-1]
        
        # Lọc dữ liệu theo khoảng thời gian
        mask = (st.session_state.data.index >= pd.to_datetime(start_date)) & (st.session_state.data.index <= pd.to_datetime(end_date))
        filtered_data = st.session_state.data.loc[mask]
        
        # Thông tin về khoảng thời gian được chọn
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%); 
                    border: 1px solid #3182ce; border-radius: 12px; padding: 16px; margin: 1rem 0;">
            <span style="color: #2c5aa0; font-weight: 600;">
                📊 Đang hiển thị {len(filtered_data):,} phiên giao dịch 
                từ {start_date.strftime('%d/%m/%Y')} đến {end_date.strftime('%d/%m/%Y')}
            </span>
            <br>
            <span style="color: #4299e1; font-weight: 400; font-size: 0.9rem; margin-top: 4px;">
                🔄 Dữ liệu VN-Index được cập nhật hàng ngày từ Sở Giao dịch Chứng khoán TP.HCM (HOSE)
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Biểu đồ giá và volume với styling chuyên nghiệp
        st.markdown('<h3 class="section-header">📈 Biểu đồ phân tích kỹ thuật</h3>', unsafe_allow_html=True)
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                'Biểu đồ nến và đường trung bình động', 
                'Khối lượng giao dịch', 
                'Chỉ số RSI (Relative Strength Index)', 
                'MACD (Moving Average Convergence Divergence)'
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Biểu đồ nến với màu sắc chuyên nghiệp
        fig.add_trace(
            go.Candlestick(
                x=filtered_data.index,
                open=filtered_data['Open'],
                high=filtered_data['High'],
                low=filtered_data['Low'],
                close=filtered_data['Close'],
                name="Giá",
                increasing_line_color='#38a169',
                decreasing_line_color='#e53e3e',
                increasing_fillcolor='#38a169',
                decreasing_fillcolor='#e53e3e'
            ),
            row=1, col=1
        )
        
        # Đường MA với màu sắc hiện đại
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data['MA_20'],
                mode='lines',
                name='MA 20',
                line=dict(color='#3182ce', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data['MA_50'],
                mode='lines',
                name='MA 50',
                line=dict(color='#d69e2e', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Volume với gradient
        colors = ['#38a169' if row['Close'] >= row['Open'] else '#e53e3e' for idx, row in filtered_data.iterrows()]
        fig.add_trace(
            go.Bar(
                x=filtered_data.index,
                y=filtered_data['Volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # RSI với zones
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='#805ad5', width=2)
            ),
            row=3, col=1
        )
        
        # RSI zones
        fig.add_hline(y=70, line_dash="dash", line_color="#e53e3e", row=3, col=1, opacity=0.7)
        fig.add_hline(y=30, line_dash="dash", line_color="#38a169", row=3, col=1, opacity=0.7)
        fig.add_hline(y=50, line_dash="dot", line_color="#718096", row=3, col=1, opacity=0.5)
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='#3182ce', width=2)
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='#e53e3e', width=2)
            ),
            row=4, col=1
        )
        
        # MACD Histogram with colors
        histogram_colors = ['#38a169' if val >= 0 else '#e53e3e' for val in filtered_data['MACD_Histogram']]
        fig.add_trace(
            go.Bar(
                x=filtered_data.index,
                y=filtered_data['MACD_Histogram'],
                name="Histogram",
                marker_color=histogram_colors,
                opacity=0.6
            ),
            row=4, col=1
        )
        
        # Layout styling
        fig.update_layout(
            height=900,
            title_text="",
            showlegend=True,
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=60, r=40, t=80, b=60),
            autosize=True
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        # Remove subplot titles styling for cleaner look
        for i in range(1, 5):
            fig.update_yaxes(title_text="", row=i, col=1)
            fig.update_xaxes(title_text="", row=i, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị dữ liệu dạng bảng với styling chuyên nghiệp
        st.markdown('<h3 class="section-header">📋 Dữ liệu giao dịch gần đây</h3>', unsafe_allow_html=True)
        
        # Tạo DataFrame hiển thị với formatting đẹp - sắp xếp theo ngày mới nhất
        display_data = filtered_data.tail(20).round(2)
        display_data = display_data.sort_index(ascending=False)
        display_data.index = display_data.index.strftime('%d/%m/%Y')
        
        st.dataframe(
            display_data[['Open', 'High', 'Low', 'Close', 'Volume']],
            use_container_width=True,
            height=400
        )
    
    with tab2:
        st.markdown('<h2 class="section-header">⚙️ Cấu hình mô hình AI</h2>', unsafe_allow_html=True)
        
        # Layout 3 cột cho cấu hình chuyên nghiệp
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <h4 style="color: #2d3748; margin-bottom: 16px; font-weight: 700; display: flex; align-items: center;">
                    Kiến trúc mô hình
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            model_type = st.selectbox(
                "🤖 Loại mô hình neural network",
                ["LSTM", "GRU"],
                help="LSTM: Phù hợp với dữ liệu chuỗi thời gian phức tạp | GRU: Tính toán nhanh hơn, hiệu quả cho dữ liệu đơn giản hơn"
            )
            
            num_neurons = st.slider(
                "🔢 Số neurons mỗi layer",
                min_value=32,
                max_value=256,
                value=64,
                step=32,
                help="Số neurons trong mỗi hidden layer. Nhiều neurons = mô hình phức tạp hơn nhưng có thể overfitting"
            )
            
            dropout_rate = st.slider(
                "🛡️ Tỷ lệ dropout",
                min_value=0.1,
                max_value=0.8,
                value=0.35,
                step=0.05,
                help="Dropout rate để tránh overfitting. 0.3-0.5 là lựa chọn phổ biến"
            )
            
            num_hidden_layers = st.slider(
                "📚 Số lớp ẩn",
                min_value=1,
                max_value=4,
                value=2,
                help="Số hidden layers trong mô hình. Nhiều layer = học được pattern phức tạp hơn"
            )
        
        with col2:
            st.markdown("""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <h4 style="color: #2d3748; margin-bottom: 16px; font-weight: 700; display: flex; align-items: center;">
                    Cấu hình huấn luyện
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            epochs = st.slider(
                "🔄 Số epochs",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Số lần huấn luyện trên toàn bộ dataset. Nhiều epochs có thể cải thiện accuracy nhưng risk overfitting"
            )
            
            batch_size = st.slider(
                "📦 Kích thước batch",
                min_value=16,
                max_value=128,
                value=32,
                step=16,
                help="Số samples xử lý cùng lúc. Batch size lớn = training nhanh hơn nhưng cần nhiều memory"
            )
            
            time_step = st.slider(
                "⏰ Số time steps",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                help="Số ngày lịch sử để dự đoán giá ngày tiếp theo. 30-60 ngày thường cho kết quả tốt"
            )
            
            validation_split = st.slider(
                "✅ Tỷ lệ validation",
                min_value=0.1,
                max_value=0.3,
                value=0.1,
                step=0.05,
                help="Phần dữ liệu dùng để validate trong quá trình training (10-20% là phổ biến)"
            )
            
            learning_rate = st.slider(
                "🎯 Learning rate",
                min_value=0.0001,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f",
                help="Tốc độ học của mô hình. Giá trị nhỏ = học chậm nhưng ổn định, giá trị lớn = học nhanh nhưng có thể không ổn định"
            )
            
            # Thêm gợi ý về learning rate
            lr_suggestions = {
                0.0001: "Rất chậm - Cho dữ liệu phức tạp",
                0.001: "Tiêu chuẩn - Lựa chọn phổ biến",
                0.003: "Nhanh - Cho thử nghiệm nhanh",
                0.01: "Rất nhanh - Có thể không ổn định"
            }
            
            closest_lr = min(lr_suggestions.keys(), key=lambda x: abs(x - learning_rate))
            if abs(closest_lr - learning_rate) < 0.0005:
                st.info(f"💡 **{closest_lr}**: {lr_suggestions[closest_lr]}")
        
        with col3:
            st.markdown("""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <h4 style="color: #2d3748; margin-bottom: 16px; font-weight: 700; display: flex; align-items: center;">
                    Cấu hình dữ liệu đầu vào
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            available_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MA_20', 'MA_50']
            
            # Feature descriptions for better UX
            feature_descriptions = {
                'Open': 'Giá mở cửa',
                'High': 'Giá cao nhất',
                'Low': 'Giá thấp nhất', 
                'Close': 'Giá đóng cửa',
                'Volume': 'Khối lượng giao dịch',
                'RSI': 'Relative Strength Index',
                'MACD': 'Moving Average Convergence Divergence',
                'MA_20': 'Đường trung bình động 20 ngày',
                'MA_50': 'Đường trung bình động 50 ngày'
            }
            
            features_to_use = st.multiselect(
                "🎯 Chọn features đầu vào",
                available_features,
                default=['Close', 'Volume', 'RSI', 'MACD'],
                help="Chọn các đặc trưng làm input cho mô hình. Close + Volume + RSI + MACD là combo phổ biến",
                format_func=lambda x: f"{x} ({feature_descriptions.get(x, x)})"
            )
            
            target_column = st.selectbox(
                "🎯 Cột dự đoán (target)",
                features_to_use if features_to_use else ['Close'],
                index=0 if 'Close' in features_to_use else 0,
                help="Cột dữ liệu mà mô hình sẽ học để dự đoán"
            )
            
            # Thêm thông tin về dataset
            if st.session_state.data is not None:
                total_samples = len(st.session_state.data) - time_step
                train_samples = int(total_samples * (1 - validation_split) * 0.8)  # 80% for train, 20% for test
                val_samples = int(total_samples * validation_split)
                test_samples = total_samples - train_samples - val_samples
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); 
                            border: 1px solid #38a169; border-radius: 12px; padding: 16px; margin-top: 1rem;">
                    <div style="color: #276749; font-weight: 600; margin-bottom: 8px;">📈 Thông tin dataset:</div>
                    <div style="color: #276749; font-size: 0.9rem;">
                        • <strong>Tổng mẫu huấn luyện:</strong> {train_samples:,} samples<br>
                        • <strong>Mẫu validation:</strong> {val_samples:,} samples<br>
                        • <strong>Mẫu test:</strong> {test_samples:,} samples<br>
                        • <strong>Features được chọn:</strong> {len(features_to_use)} features<br>
                        • <strong>Sequence length:</strong> {time_step} time steps
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Hiển thị tổng quan cấu hình
        if features_to_use and target_column:
            st.markdown('<h3 class="section-header">📋 Tổng quan cấu hình mô hình</h3>', unsafe_allow_html=True)
            
            config_summary_cols = st.columns(4)
            
            with config_summary_cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Kiến trúc</div>
                    <div class="metric-value" style="font-size: 1.8rem;">{model_type}</div>
                    <div class="metric-sublabel">{num_neurons} neurons × {num_hidden_layers} layers</div>
                </div>
                """, unsafe_allow_html=True)
            
            with config_summary_cols[1]:
                estimated_time = epochs * 2 if model_type == "LSTM" else epochs * 1.5  # Rough estimation
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Training Config</div>
                    <div class="metric-value" style="font-size: 1.8rem;">{epochs}</div>
                    <div class="metric-sublabel">epochs, LR: {learning_rate:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with config_summary_cols[2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Input Shape</div>
                    <div class="metric-value" style="font-size: 1.8rem;">{len(features_to_use)}</div>
                    <div class="metric-sublabel">{time_step} × {len(features_to_use)} features</div>
                </div>
                """, unsafe_allow_html=True)
            
            with config_summary_cols[3]:
                complexity_score = (num_neurons * num_hidden_layers * len(features_to_use)) / 1000
                complexity_level = "Thấp" if complexity_score < 1 else "Trung bình" if complexity_score < 5 else "Cao"
                complexity_color = "#38a169" if complexity_score < 1 else "#d69e2e" if complexity_score < 5 else "#e53e3e"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Độ phức tạp</div>
                    <div class="metric-value" style="font-size: 1.8rem; color: {complexity_color};">{complexity_level}</div>
                    <div class="metric-sublabel">Score: {complexity_score:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Success message with next step guidance
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); 
                        border-left: 6px solid #38a169; border-radius: 12px; padding: 20px; margin-top: 1.5rem;">
                <div style="color: #276749; font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;">
                    ✅ Cấu hình hoàn tất!
                </div>
                <div style="color: #276749; font-size: 0.95rem;">
                    🚀 Mô hình đã sẵn sàng để huấn luyện. Chuyển sang tab <strong>"🧪 Huấn luyện"</strong> để bắt đầu.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fffaf0 0%, #fbd38d 100%); 
                        border-left: 6px solid #d69e2e; border-radius: 12px; padding: 20px; margin-top: 1.5rem;">
                <div style="color: #b7791f; font-weight: 600; font-size: 1rem;">
                    ⚠️ Cấu hình chưa hoàn tất
                </div>
                <div style="color: #b7791f; font-size: 0.9rem; margin-top: 4px;">
                    Vui lòng chọn ít nhất một feature và target column để tiếp tục.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Lưu cấu hình vào session state
        st.session_state.model_config = {
            'model_type': model_type,
            'num_neurons': num_neurons,
            'dropout_rate': dropout_rate,
            'num_hidden_layers': num_hidden_layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'time_step': time_step,
            'validation_split': validation_split,
            'learning_rate': learning_rate,
            'features_to_use': features_to_use,
            'target_column': target_column
        }
        
        if features_to_use and target_column:
            st.success(f"✅ Cấu hình hoàn tất! Sẵn sàng huấn luyện mô hình {model_type}")
        else:
            st.warning("⚠️ Vui lòng chọn ít nhất một feature và target column")
    
    with tab3:
        st.header("🧪 Huấn luyện mô hình")
        
        if 'model_config' in st.session_state and st.session_state.model_config['features_to_use']:
            config = st.session_state.model_config
            
            # Hiển thị cấu hình hiện tại
            st.subheader("Cấu hình hiện tại")
            
            # Hiển thị cấu hình chi tiết trong 3 cột
            config_cols = st.columns(3)
            
            with config_cols[0]:
                st.info("**Thông số mô hình**")
                st.write(f"- **Loại mô hình:** {config['model_type']}")
                st.write(f"- **Số neurons:** {config['num_neurons']}")
                st.write(f"- **Dropout rate:** {config['dropout_rate']}")
                st.write(f"- **Số hidden layers:** {config['num_hidden_layers']}")
            
            with config_cols[1]:
                st.info("**Thông số huấn luyện**")
                st.write(f"- **Epochs:** {config['epochs']}")
                st.write(f"- **Batch size:** {config['batch_size']}")
                st.write(f"- **Time steps:** {config['time_step']}")
                st.write(f"- **Validation split:** {config['validation_split']}")
                st.write(f"- **Learning rate:** {config.get('learning_rate', 0.001)}")
            
            with config_cols[2]:
                st.info("**Thông số dữ liệu**")
                st.write(f"- **Features:** {len(config['features_to_use'])}")
                st.write(f"- **Features đầu vào:** {', '.join(config['features_to_use'])}")
                st.write(f"- **Target:** {config['target_column']}")
                if 'data' in st.session_state and st.session_state.data is not None:
                    st.write(f"- **Số dòng dữ liệu:** {len(st.session_state.data)}")

            # Cảnh báo không chuyển tab khi huấn luyện
            st.warning("⚠️ Khi huấn luyện, vui lòng không chuyển tab hoặc thao tác khác cho đến khi hoàn tất để tránh lỗi trạng thái!", icon="⚠️")
            
            # Nút huấn luyện
            if st.button("🚀 Bắt đầu huấn luyện", type="primary", use_container_width=True):
                try:
                    with st.spinner("Đang tiền xử lý dữ liệu..."):
                        # Tiền xử lý dữ liệu
                        X_train, y_train, X_test, y_test, scaler, scaled_data = preprocess_data(
                            st.session_state.data.dropna(),
                            config['features_to_use'],
                            config['target_column'],
                            config['time_step']
                        )
                        
                        st.session_state.scaler = scaler
                        st.session_state.scaled_data = scaled_data
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        # Ghi thông tin cấu hình mô hình ra file JSON để minh bạch tái lập
                        try:
                            # Chuẩn bị thông tin dataset
                            data_df = st.session_state.data
                            date_min = data_df.index.min().date().isoformat() if len(data_df.index) else None
                            date_max = data_df.index.max().date().isoformat() if len(data_df.index) else None

                            # Kiểm tra tính dừng cho ARIMA (trên cột target)
                            try:
                                series_for_arima, stationarity = prepare_data_for_arima(data_df, target_column=config['target_column'])
                                stationarity_info = {
                                    "adf_stat": round(float(stationarity.get('adf_statistic', float('nan'))), 4) if stationarity else None,
                                    "p_value": round(float(stationarity.get('p_value', float('nan'))), 4) if stationarity else None,
                                    "stationary": bool(stationarity.get('is_stationary')) if stationarity else None
                                }
                            except Exception:
                                stationarity_info = {
                                    "adf_stat": None,
                                    "p_value": None,
                                    "stationary": None
                                }

                            # Mô tả các layer của NN hiện tại
                            nn_layers = [{
                                "type": config['model_type'],
                                "units": int(config['num_neurons']),
                                "dropout": float(config['dropout_rate'])
                            } for _ in range(int(config['num_hidden_layers']))]
                            nn_layers.append({
                                "type": "Dense",
                                "units": 1,
                                "activation": "linear"
                            })

                            # Xây dựng object JSON theo mẫu cung cấp
                            model_info = {
                                "dataset": {
                                    "name": "VNIndex",
                                    "features": list(config['features_to_use']),
                                    "sequence_length": int(config['time_step']),
                                    "train_split": 0.8,
                                    "date_range": {
                                        "from": date_min,
                                        "to": date_max
                                    }
                                },
                                config['model_type']: {
                                    "layers": nn_layers,
                                    "epochs": int(config['epochs']),
                                    "batch_size": int(config['batch_size']),
                                    "learning_rate": float(config.get('learning_rate', 0.001)),
                                    "validation_split": float(config.get('validation_split', 0.1)),
                                    "optimizer": "adam",
                                    "loss": "mae"
                                },
                                "ARIMA": {
                                    "order": None,
                                    "seasonal_order": None,
                                    "stationarity_test": stationarity_info,
                                    "auto_param_search": False
                                },
                                "meta": {
                                    "created_at": datetime.utcnow().isoformat() + "Z",
                                    "created_by": os.getenv("USER", "dashboard"),
                                    "reproducibility": {"seed": 42},
                                    "notes": "Generated automatically when training starts."
                                }
                            }

                            # Nếu model hiện tại không phải GRU/LSTM còn lại, thêm mục 'planned' đơn giản cho loại kia
                            other_type = "GRU" if config['model_type'] == "LSTM" else "LSTM"
                            model_info[other_type] = {"planned": True}

                            os.makedirs("models", exist_ok=True)
                            with open(os.path.join("models", "model_info.json"), "w", encoding="utf-8") as f:
                                json.dump(model_info, f, ensure_ascii=False, indent=2)
                        except Exception as _json_err:
                            # Không chặn quá trình train nếu ghi file lỗi, chỉ thông báo nhẹ
                            st.warning(f"Không thể lưu model_info.json: {_json_err}")
                    
                    st.success(f"✅ Tiền xử lý hoàn tất! Shape: Train {X_train.shape}, Test {X_test.shape}")
                    
                    with st.spinner("Đang xây dựng và huấn luyện mô hình..."):
                        # Xây dựng mô hình
                        model = build_model(
                            config['model_type'],
                            config['time_step'],
                            len(config['features_to_use']),
                            config['num_neurons'],
                            config['dropout_rate'],
                            config['num_hidden_layers'],
                            learning_rate=config.get('learning_rate', 0.001)
                        )
                        
                        # Callback
                        early_stopping = EarlyStopping(
                            monitor='val_loss',
                            patience=10,
                            min_delta=0.001,
                            restore_best_weights=True
                        )
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Custom callback để cập nhật progress
                        class StreamlitCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress = (epoch + 1) / config['epochs']
                                progress_bar.progress(progress)
                                status_text.text(f"Epoch {epoch + 1}/{config['epochs']} - Loss: {logs['loss']:.4f} - Val Loss: {logs['val_loss']:.4f}")
                        
                        # Huấn luyện
                        history = train_model(
                            model,
                            X_train, y_train,
                            config,
                            callbacks=[early_stopping, StreamlitCallback()]
                        )
                        
                        st.session_state.model = model
                        st.session_state.training_history = history
                        
                        # Đánh giá mô hình (trong cùng khối spinner)
                        metrics, y_test_inv, y_pred_inv = evaluate_model(
                            model, X_test, y_test, scaler,
                            config['features_to_use'].index(config['target_column'])
                        )
                        
                        st.session_state.y_test_inv = y_test_inv
                        st.session_state.y_pred_inv = y_pred_inv
                        st.session_state.model_metrics = metrics
                    
                    st.success("🎉 Huấn luyện và đánh giá hoàn tất!")
                    
                    # Hiển thị kết quả (chỉ khi có dữ liệu)
                    if (st.session_state.model_metrics and 
                        st.session_state.training_history and 
                        st.session_state.y_test_inv is not None and 
                        st.session_state.y_pred_inv is not None):
                        
                        st.subheader("📊 Kết quả đánh giá")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("MAE", f"{st.session_state.model_metrics['MAE']:.2f}")
                        
                        with col2:
                            st.metric("MSE", f"{st.session_state.model_metrics['MSE']:.2f}")
                        
                        with col3:
                            st.metric("RMSE", f"{st.session_state.model_metrics['RMSE']:.2f}")
                        
                        with col4:
                            st.metric("R²", f"{st.session_state.model_metrics['R²']:.4f}")
                        
                        # Biểu đồ loss
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=st.session_state.training_history.history['loss'],
                            mode='lines',
                            name='Training Loss'
                        ))
                        fig_loss.add_trace(go.Scatter(
                            y=st.session_state.training_history.history['val_loss'],
                            mode='lines',
                            name='Validation Loss'
                        ))
                        fig_loss.update_layout(
                            title="Training History",
                            xaxis_title="Epoch",
                            yaxis_title="Loss"
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                        
                        # Biểu đồ so sánh dự đoán
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(
                            y=st.session_state.y_test_inv[-100:],
                            mode='lines',
                            name='Thực tế',
                            line=dict(color='blue')
                        ))
                        fig_pred.add_trace(go.Scatter(
                            y=st.session_state.y_pred_inv[-100:],
                            mode='lines',
                            name='Dự đoán',
                            line=dict(color='red')
                        ))
                        fig_pred.update_layout(
                            title="So sánh dự đoán vs thực tế (100 ngày cuối)",
                            xaxis_title="Ngày",
                            yaxis_title="Giá"
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Lỗi trong quá trình huấn luyện: {e}")
        else:
            st.warning("⚠️ Vui lòng cấu hình mô hình trước khi huấn luyện!")
    
    with tab4:
        st.header("📈 Dự đoán tương lai")
        
        if st.session_state.model is not None and st.session_state.scaler is not None:
            st.subheader("Cấu hình dự đoán")

            n_future = st.slider(
                "Số ngày dự đoán",
                min_value=1,
                max_value=20,
                value=5,
                help="Số ngày muốn dự đoán trong tương lai"
            )

            if st.button("🔮 Dự đoán tương lai", type="primary"):
                try:
                    with st.spinner("Đang thực hiện dự đoán..."):
                        config = st.session_state.model_config

                        # Thêm progress bar cho dự đoán nhiều ngày
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Thực hiện dự đoán từng ngày
                        future_predictions = []
                        # Tạo future_dates chỉ gồm ngày giao dịch (thứ 2-6)
                        last_date = st.session_state.data.index[-1]
                        future_dates = []
                        days_added = 0
                        current_date = last_date
                        while days_added < n_future:
                            current_date += pd.Timedelta(days=1)
                            if current_date.weekday() < 5:  # 0=Mon, ..., 4=Fri
                                future_dates.append(current_date)
                                days_added += 1

                        for i in range(n_future):
                            preds = predict_future(
                                st.session_state.model,
                                st.session_state.scaled_data,
                                config['time_step'],
                                i+1,
                                st.session_state.scaler,
                                len(config['features_to_use']),
                                config['features_to_use'].index(config['target_column']),
                                config['features_to_use']
                            )
                            # Lấy giá trị dự đoán mới nhất
                            future_predictions.append(preds[-1])
                            progress_bar.progress((i+1)/n_future)
                            status_text.text(f"Dự đoán ngày {i+1}/{n_future}")
                        st.session_state.future_predictions = future_predictions
                        progress_bar.empty()
                        status_text.empty()
                    st.success(f"✅ Dự đoán {n_future} ngày tương lai hoàn tất!")

                    # Hiển thị kết quả dự đoán
                    st.subheader("📊 Kết quả dự đoán")

                    # Tạo DataFrame cho dự đoán
                    prediction_df = pd.DataFrame({
                        'Ngày': future_dates,
                        'Giá dự đoán': future_predictions
                    })

                    # Hiển thị bảng dự đoán
                    st.dataframe(prediction_df, use_container_width=True)

                    # Biểu đồ dự đoán
                    fig = go.Figure()

                    # Dữ liệu lịch sử (100 ngày cuối)
                    historical_data = st.session_state.data.tail(100)
                    fig.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=historical_data['Close'],
                        mode='lines',
                        name='Dữ liệu lịch sử',
                        line=dict(color='blue')
                    ))

                    # Dự đoán tương lai
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_predictions,
                        mode='lines+markers',
                        name='Dự đoán tương lai',
                        line=dict(color='red', dash='dash'),
                        marker=dict(size=6)
                    ))

                    # Đường nối giữa dữ liệu lịch sử và dự đoán
                    fig.add_trace(go.Scatter(
                        x=[historical_data.index[-1], future_dates[0]],
                        y=[historical_data['Close'].iloc[-1], future_predictions[0]],
                        mode='lines',
                        name='Kết nối',
                        line=dict(color='gray', dash='dot'),
                        showlegend=False
                    ))

                    fig.update_layout(
                        title=f"Dự đoán giá VNIndex cho {n_future} ngày tới",
                        xaxis_title="Ngày",
                        yaxis_title="Giá",
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Thống kê dự đoán
                    col1, col2, col3, col4 = st.columns(4)

                    current_price = st.session_state.data['Close'].iloc[-1]
                    predicted_price = future_predictions[-1]
                    price_change = predicted_price - current_price
                    price_change_pct = (price_change / current_price) * 100

                    with col1:
                        st.metric("Giá hiện tại", f"{current_price:,.0f}")

                    with col2:
                        st.metric("Giá dự đoán", f"{predicted_price:,.0f}")

                    with col3:
                        st.metric("Thay đổi", f"{price_change:,.0f}", delta=f"{price_change:,.0f}")

                    with col4:
                        st.metric("Thay đổi (%)", f"{price_change_pct:.2f}%", delta=f"{price_change_pct:.2f}%")
                except Exception as e:
                    st.error(f"❌ Lỗi trong quá trình dự đoán: {e}")
        else:
            st.warning("⚠️ Vui lòng huấn luyện mô hình trước khi dự đoán!")
    
    with tab5:
        st.header("🔄 So sánh mô hình LSTM/GRU vs ARIMA")
        
        if st.session_state.data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚙️ Cấu hình ARIMA")
                
                # Kiểm tra tính dừng
                series, stationarity_test = prepare_data_for_arima(st.session_state.data)
                
                st.write("**Kiểm tra tính dừng (ADF Test):**")
                st.write(f"- ADF Statistic: {stationarity_test['adf_statistic']:.4f}")
                st.write(f"- P-value: {stationarity_test['p_value']:.4f}")
                st.write(f"- Chuỗi dừng: {'✅ Có' if stationarity_test['is_stationary'] else '❌ Không'}")
                
                # Tùy chọn tham số ARIMA
                auto_arima = st.checkbox("Tự động tìm tham số tối ưu", value=True)
                
                if not auto_arima:
                    col_p, col_d, col_q = st.columns(3)
                    with col_p:
                        p = st.number_input("p (AR order)", min_value=0, max_value=5, value=1)
                    with col_d:
                        d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
                    with col_q:
                        q = st.number_input("q (MA order)", min_value=0, max_value=5, value=1)
                    arima_order = (p, d, q)
                else:
                    arima_order = None
                
                # Tỷ lệ chia dữ liệu
                train_ratio = st.slider("Tỷ lệ dữ liệu huấn luyện (%)", 60, 90, 80) / 100
                
            with col2:
                st.subheader("📊 Kết quả so sánh")
                
                if st.button("🚀 Chạy so sánh mô hình"):
                    with st.spinner("Đang huấn luyện và so sánh mô hình..."):
                        try:
                            # Chia dữ liệu
                            train_size = int(len(series) * train_ratio)
                            train_data = series[:train_size]
                            test_data = series[train_size:]
                            
                            # Huấn luyện ARIMA
                            arima_model, final_order = train_arima_model(train_data, arima_order)
                            
                            if arima_model is not None:
                                st.success(f"✅ ARIMA{final_order} huấn luyện thành công!")
                                
                                # Dự đoán ARIMA
                                arima_predictions = predict_arima(arima_model, len(test_data))
                                
                                if arima_predictions is not None:
                                    # Đánh giá ARIMA
                                    arima_metrics = evaluate_arima_model(test_data.values, arima_predictions)
                                    
                                    # Hiển thị kết quả ARIMA
                                    st.write("**Hiệu suất ARIMA:**")
                                    col_mae, col_mse, col_rmse, col_r2 = st.columns(4)
                                    
                                    with col_mae:
                                        st.metric("MAE", f"{arima_metrics['MAE']:.4f}")
                                    with col_mse:
                                        st.metric("MSE", f"{arima_metrics['MSE']:.4f}")
                                    with col_rmse:
                                        st.metric("RMSE", f"{arima_metrics['RMSE']:.4f}")
                                    with col_r2:
                                        st.metric("R²", f"{arima_metrics['R²']:.4f}")
                                    
                                    # Lưu kết quả ARIMA vào session state
                                    st.session_state.arima_model = arima_model
                                    st.session_state.arima_metrics = arima_metrics
                                    st.session_state.arima_predictions = arima_predictions
                                    st.session_state.arima_test_data = test_data
                                    
                                    # So sánh với LSTM/GRU nếu có
                                    if hasattr(st.session_state, 'model_metrics') and st.session_state.model_metrics:
                                        comparison = compare_models_performance(
                                            st.session_state.model_metrics, 
                                            arima_metrics
                                        )
                                        
                                        st.write("**So sánh LSTM/GRU vs ARIMA:**")
                                        comparison_df = pd.DataFrame({
                                            'Metric': list(comparison.keys()),
                                            'LSTM/GRU': [comparison[m]['LSTM/GRU'] for m in comparison.keys()],
                                            'ARIMA': [comparison[m]['ARIMA'] for m in comparison.keys()],
                                            'Mô hình tốt hơn': [comparison[m]['better_model'] for m in comparison.keys()]
                                        })
                                        st.dataframe(comparison_df, use_container_width=True)
                                    else:
                                        st.info("💡 Huấn luyện mô hình LSTM/GRU trước để so sánh!")
                                else:
                                    st.error("❌ Lỗi khi dự đoán với ARIMA")
                            else:
                                st.error("❌ Lỗi khi huấn luyện ARIMA")
                                
                        except Exception as e:
                            st.error(f"❌ Lỗi: {str(e)}")
            
            # Biểu đồ so sánh dự đoán
            if (hasattr(st.session_state, 'arima_predictions') and 
                hasattr(st.session_state, 'arima_test_data')):
                
                st.subheader("📈 Biểu đồ so sánh dự đoán")
                
                # Tạo biểu đồ so sánh
                fig = go.Figure()
                
                # Dữ liệu thực tế
                fig.add_trace(go.Scatter(
                    x=st.session_state.arima_test_data.index,
                    y=st.session_state.arima_test_data.values,
                    mode='lines',
                    name='Giá thực tế',
                    line=dict(color='blue', width=2)
                ))
                
                # Dự đoán ARIMA
                fig.add_trace(go.Scatter(
                    x=st.session_state.arima_test_data.index,
                    y=st.session_state.arima_predictions,
                    mode='lines',
                    name='Dự đoán ARIMA',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Dự đoán LSTM/GRU nếu có
                if (hasattr(st.session_state, 'test_predictions') and 
                    st.session_state.test_predictions is not None):
                    fig.add_trace(go.Scatter(
                        x=st.session_state.arima_test_data.index[-len(st.session_state.test_predictions):],
                        y=st.session_state.test_predictions.flatten(),
                        mode='lines',
                        name='Dự đoán LSTM/GRU',
                        line=dict(color='green', width=2, dash='dot')
                    ))
                
                fig.update_layout(
                    title="So sánh dự đoán giữa các mô hình",
                    xaxis_title="Thời gian",
                    yaxis_title="Giá",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Vui lòng tải dữ liệu trước!")

    with tab6:
        st.header("💾 Quản lý mô hình")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Lưu mô hình")
            
            if st.session_state.model is not None:
                model_name = st.text_input(
                    "Tên mô hình",
                    value=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                if st.button("💾 Lưu mô hình", type="primary"):
                    try:
                        # Tạo thư mục models nếu chưa có và đảm bảo quyền
                        models_dir = "models"
                        if not os.path.exists(models_dir):
                            os.makedirs(models_dir, exist_ok=True)
                            os.chmod(models_dir, 0o777) # Đặt quyền đầy đủ
                        
                        # Lưu mô hình
                        model_path = f"models/{model_name}.h5"
                        st.session_state.model.save(model_path)
                        
                        # Lưu scaler và config
                        with open(f"models/{model_name}_scaler.pkl", 'wb') as f:
                            pickle.dump(st.session_state.scaler, f)
                        
                        with open(f"models/{model_name}_config.pkl", 'wb') as f:
                            pickle.dump(st.session_state.model_config, f)
                        
                        st.success(f"✅ Đã lưu mô hình: {model_name}")
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi khi lưu mô hình: {e}")
            else:
                st.warning("⚠️ Chưa có mô hình để lưu!")
        
        with col2:
            st.subheader("Tải mô hình")
            
            # Liệt kê các mô hình đã lưu
            if os.path.exists("models"):
                model_files = [f for f in os.listdir("models") if f.endswith(".h5")]
                
                if model_files:
                    selected_model_load = st.selectbox(
                        "Chọn mô hình để tải",
                        model_files,
                        key="load_model_selectbox"
                    )
                    
                    if st.button("📂 Tải mô hình"):
                        try:
                            model_name = selected_model_load.replace(".h5", ".h5")
                            
                            # Tải mô hình với custom_objects
                            st.session_state.model = keras_load_model(
                                f"models/{model_name}",
                                custom_objects={
                                    'mae': tf.keras.metrics.MeanAbsoluteError(),
                                    'mse': tf.keras.metrics.MeanSquaredError()
                                }
                            )
                            
                            # Tải scaler
                            with open(f"models/{model_name.replace('.h5', '')}_scaler.pkl", 'rb') as f:
                                st.session_state.scaler = pickle.load(f)
                            
                            # Tải config
                            with open(f"models/{model_name.replace('.h5', '')}_config.pkl", 'rb') as f:
                                st.session_state.model_config = pickle.load(f)
                            
                            # Tính lại scaled_data để sử dụng cho predict_future
                            if st.session_state.data is not None:
                                try:
                                    df = st.session_state.data.dropna()
                                    features = st.session_state.model_config['features_to_use']
                                    # Chuẩn hóa theo scaler đã load
                                    data_to_scale = df[features].values
                                    scaled_data = st.session_state.scaler.transform(data_to_scale)
                                    st.session_state.scaled_data = scaled_data
                                    st.success(f"✅ Đã tải mô hình: {model_name.replace('.h5', '')} - Sẵn sàng dự đoán!")
                                    st.rerun()  # Force refresh giao diện để hiển thị button dự đoán
                                except Exception as e:
                                    st.success(f"✅ Đã tải mô hình: {model_name.replace('.h5', '')}")
                                    st.warning(f"⚠️ Không thể tính lại dữ liệu đã scale: {e}")
                                    st.rerun()  # Force refresh ngay cả khi có lỗi scale
                            else:
                                st.success(f"✅ Đã tải mô hình: {model_name.replace('.h5', '')}")
                                st.warning("⚠️ Cần tải dữ liệu trước để sử dụng dự đoán")
                                st.rerun()  # Force refresh để hiển thị UI đã cập nhật
                            
                        except Exception as e:
                            st.error(f"❌ Lỗi khi tải mô hình: {e}")
                            st.rerun()  # Refresh để đảm bảo UI consistent
                else:
                    st.info("📁 Chưa có mô hình nào được lưu")
            else:
                st.info("📁 Thư mục models chưa tồn tại")

        with col2:
            st.subheader("Xóa mô hình")
            if os.path.exists("models"):
                model_files_to_delete = [f for f in os.listdir("models") if f.endswith(".h5")]
                if model_files_to_delete:
                    selected_model_delete = st.selectbox(
                        "Chọn mô hình để xóa",
                        model_files_to_delete,
                        key="delete_model_selectbox"
                    )
                    if st.button("🗑️ Xóa mô hình", type="secondary"):
                        try:
                            model_base_name = selected_model_delete.replace(".h5", "")
                            model_path = f"models/{model_base_name}.h5"
                            scaler_path = f"models/{model_base_name}_scaler.pkl"
                            config_path = f"models/{model_base_name}_config.pkl"

                            if os.path.exists(model_path):
                                os.remove(model_path)
                            if os.path.exists(scaler_path):
                                os.remove(scaler_path)
                            if os.path.exists(config_path):
                                os.remove(config_path)
                            st.success(f"✅ Đã xóa mô hình: {model_base_name}")
                            # Attempt to refresh the page; wrap in case the method is unavailable
                            try:
                                st.experimental_rerun()
                            except AttributeError:
                                pass
                        except Exception as e:
                            st.error(f"❌ Lỗi khi xóa mô hình: {e}")
                else:
                    st.info("📁 Không có mô hình nào để xóa")
            else:
                st.info("📁 Thư mục models chưa tồn tại")

else:
    # Landing page khi chưa có dữ liệu
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fffaf0 0%, #fbd38d 100%); 
                border-left: 6px solid #d69e2e; border-radius: 16px; padding: 24px; margin: 2rem 0;">
        <div style="color: #b7791f; font-weight: 700; font-size: 1.2rem; margin-bottom: 12px;">
            ⚠️ Chưa có dữ liệu để phân tích
        </div>
        <div style="color: #b7791f; font-size: 1rem;">
            Vui lòng tải dữ liệu từ sidebar để bắt đầu sử dụng dashboard.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hướng dẫn sử dụng với thiết kế chuyên nghiệp
    st.markdown('<h2 class="section-header">📖 Hướng dẫn sử dụng Dashboard</h2>', unsafe_allow_html=True)
    
    # Workflow steps
    workflow_steps = [
        {
            "icon": "📁",
            "title": "Tải dữ liệu",
            "description": "Upload file CSV hoặc sử dụng dữ liệu mặc định từ sidebar",
            "color": "#3182ce"
        },
        {
            "icon": "📊", 
            "title": "Phân tích dữ liệu",
            "description": "Xem biểu đồ, chỉ báo kỹ thuật và thống kê thị trường",
            "color": "#38a169"
        },
        {
            "icon": "⚙️",
            "title": "Cấu hình mô hình",
            "description": "Thiết lập tham số LSTM/GRU và chọn features",
            "color": "#d69e2e"
        },
        {
            "icon": "🧪",
            "title": "Huấn luyện AI",
            "description": "Train mô hình với dữ liệu và đánh giá hiệu suất",
            "color": "#805ad5"
        },
        {
            "icon": "📈",
            "title": "Dự đoán tương lai",
            "description": "Dự đoán giá cổ phiếu và phân tích xu hướng",
            "color": "#e53e3e"
        },
        {
            "icon": "💾",
            "title": "Quản lý mô hình",
            "description": "Lưu, tải và quản lý các mô hình đã huấn luyện",
            "color": "#718096"
        }
    ]
    
    # Hiển thị workflow trong grid
    cols = st.columns(3)
    for i, step in enumerate(workflow_steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; min-height: 180px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">{step['icon']}</div>
                <div style="font-weight: 700; font-size: 1.1rem; color: {step['color']}; margin-bottom: 8px;">
                    {step['title']}
                </div>
                <div style="color: #718096; font-size: 0.9rem; line-height: 1.4;">
                    {step['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Technical requirements
    st.markdown('<h3 class="section-header">📊 Yêu cầu định dạng dữ liệu</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #2d3748; margin-bottom: 16px; font-weight: 700;">📋 Cấu trúc file CSV</h4>
            <div style="color: #4a5568; font-size: 0.9rem; line-height: 1.6;">
                File CSV cần có các cột bắt buộc:<br><br>
                • <strong>Date</strong>: Ngày giao dịch (YYYY-MM-DD)<br>
                • <strong>Open</strong>: Giá mở cửa<br>
                • <strong>High</strong>: Giá cao nhất<br>
                • <strong>Low</strong>: Giá thấp nhất<br>
                • <strong>Close</strong>: Giá đóng cửa<br>
                • <strong>Volume</strong>: Khối lượng giao dịch
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #2d3748; margin-bottom: 16px; font-weight: 700;">⚡ Tính năng nâng cao</h4>
            <div style="color: #4a5568; font-size: 0.9rem; line-height: 1.6;">
                Dashboard tự động tính toán:<br><br>
                • <strong>RSI</strong>: Relative Strength Index<br>
                • <strong>MACD</strong>: Moving Average Convergence Divergence<br>
                • <strong>MA</strong>: Đường trung bình động 20/50 ngày<br>
                • <strong>Bollinger Bands</strong>: Dải giá dao động<br>
                • <strong>Volume Analysis</strong>: Phân tích khối lượng<br>
                • <strong>Technical Indicators</strong>: Các chỉ báo kỹ thuật
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data preview
    st.markdown('<h3 class="section-header">👀 Ví dụ dữ liệu mẫu</h3>', unsafe_allow_html=True)
    
    sample_data = {
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05'],
        'Open': [1250.5, 1255.2, 1260.1, 1258.8, 1262.3],
        'High': [1268.4, 1272.1, 1275.6, 1271.2, 1278.9],
        'Low': [1248.1, 1251.8, 1257.3, 1255.4, 1259.7],
        'Close': [1265.8, 1269.5, 1271.2, 1268.1, 1275.4],
        'Volume': [125000000, 132000000, 118000000, 145000000, 139000000]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
    
    # Call to action
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%); 
                border-radius: 16px; padding: 32px; text-align: center; margin: 2rem 0;">
        <div style="font-size: 1.5rem; font-weight: 700; color: #2c5aa0; margin-bottom: 12px;">
            🚀 Sẵn sàng bắt đầu?
        </div>
        <div style="color: #2c5aa0; font-size: 1.1rem; margin-bottom: 16px;">
            Tải dữ liệu lên từ sidebar để khám phá sức mạnh của AI trong dự đoán chứng khoán!
        </div>
        <div style="font-size: 0.9rem; color: #4299e1;">
            💡 Tip: Sử dụng dữ liệu ít nhất 1 năm để có kết quả dự đoán tốt nhất
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer chuyên nghiệp
st.markdown("""
<div class="footer">
    <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.2rem;">📈</span>
                <span style="font-weight: 600; color: #2d3748;">VNIndex Prediction Dashboard</span>
            </div>
            <div style="font-size: 0.9rem; color: #718096;">
                Powered by <strong>Streamlit</strong> • <strong>TensorFlow</strong> • <strong>Plotly</strong>
            </div>
            <div style="font-size: 0.85rem; color: #a0aec0;">
                © 2025 AI-Powered Financial Analytics
            </div>
        </div>
        <div style="text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; color: #a0aec0; font-size: 0.8rem;">
            ⚠️ Thông tin chỉ mang tính chất tham khảo. Không phải lời khuyên đầu tư.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
