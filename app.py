# app.py
import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from tensorflow import keras
import pickle
import time
from pathlib import Path


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: #0a0a0f;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0.75rem;
        font-weight: 400;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        color: #3b82f6;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, #12121a 0%, #1a1a28 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }
    
    .metric-card h2 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-card p {
        color: #64748b;
        font-size: 0.875rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Result cards */
    .result-card {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-card-real {
        background: linear-gradient(145deg, #052e16 0%, #14532d 100%);
        border: 1px solid #22c55e;
        box-shadow: 0 0 60px rgba(34, 197, 94, 0.15);
    }
    
    .result-card-fake {
        background: linear-gradient(145deg, #450a0a 0%, #7f1d1d 100%);
        border: 1px solid #ef4444;
        box-shadow: 0 0 60px rgba(239, 68, 68, 0.15);
    }
    
    .result-icon {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem auto;
        font-size: 1.5rem;
    }
    
    .result-icon-real {
        background: rgba(34, 197, 94, 0.2);
        border: 2px solid #22c55e;
    }
    
    .result-icon-fake {
        background: rgba(239, 68, 68, 0.2);
        border: 2px solid #ef4444;
    }
    
    .result-label {
        font-size: 1.75rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: 0.5px;
    }
    
    .confidence-text {
        font-size: 1rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .confidence-bar {
        height: 6px;
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
        margin-top: 1.5rem;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    
    .confidence-fill-real {
        background: linear-gradient(90deg, #22c55e, #4ade80);
    }
    
    .confidence-fill-fake {
        background: linear-gradient(90deg, #ef4444, #f87171);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0a0a0f;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background: #12121a !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #475569 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary button */
    .secondary-btn > button {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
        box-shadow: none !important;
    }
    
    .secondary-btn > button:hover {
        background: rgba(255,255,255,0.05) !important;
        border-color: rgba(255,255,255,0.2) !important;
    }
    
    /* Model selector card */
    .model-selector {
        background: #12121a;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .model-selector:hover {
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .model-selector h4 {
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0 0 0.25rem 0;
    }
    
    .model-selector p {
        color: #64748b;
        font-size: 0.8rem;
        margin: 0;
        line-height: 1.4;
    }
    
    .model-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    .badge-transformer {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
    }
    
    .badge-lstm {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
    }
    
    /* Info box */
    .info-box {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.2);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .info-box p {
        color: #94a3b8;
        margin: 0;
        font-size: 0.875rem;
        line-height: 1.5;
    }
    
    .info-box strong {
        color: #e2e8f0;
    }
    
    /* Warning box */
    .warning-box {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.2);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .warning-box p {
        color: #fbbf24;
        margin: 0;
        font-size: 0.875rem;
    }
    
    /* Tips card */
    .tips-card {
        background: #12121a;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1.5rem;
    }
    
    .tips-card h3 {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }
    
    .tip-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
        color: #94a3b8;
    }
    
    .tip-icon {
        width: 20px;
        height: 20px;
        background: rgba(34, 197, 94, 0.15);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.75rem;
        flex-shrink: 0;
        color: #22c55e;
        font-size: 0.7rem;
    }
    
    /* Section title */
    .section-title {
        color: #e2e8f0;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(255,255,255,0.1);
        margin-left: 1rem;
    }
    
    /* Stats row */
    .stats-row {
        display: flex;
        gap: 0.5rem;
        margin: 0.75rem 0 1.5rem 0;
    }
    
    .stat-item {
        background: rgba(255,255,255,0.03);
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-size: 0.8rem;
        color: #64748b;
    }
    
    .stat-item strong {
        color: #94a3b8;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #12121a;
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #64748b;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
    }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border-radius: 4px;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: rgba(255,255,255,0.05);
        margin: 2rem 0;
    }
    
    /* Device badge */
    .device-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.2);
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-size: 0.8rem;
        color: #22c55e;
    }
    
    .device-badge-cpu {
        background: rgba(245, 158, 11, 0.1);
        border-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
    }
    
    /* Comparison table */
    .comparison-table {
        background: #12121a;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Consensus box */
    .consensus-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1.5rem;
    }
    
    .consensus-box h3 {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
    }
    
    .consensus-result {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .consensus-real {
        color: #22c55e;
    }
    
    .consensus-fake {
        color: #ef4444;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0f;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL CONFIGURATION ====================
MODEL_CONFIGS = {
    "BERT": {
        "path": "bert_model",
        "type": "transformer",
        "name": "BERT",
        "description": "Bidirectional Encoder Representations from Transformers - High accuracy model",
        "color": "#3b82f6"
    },
    "DistilBERT": {
        "path": "distilbert_model",
        "type": "transformer",
        "name": "DistilBERT",
        "description": "Lighter and faster version of BERT with comparable performance",
        "color": "#22c55e"
    },
    "LSTM": {
        "path": "lstm_model.keras",
        "tokenizer_path": "lstm_tokenizer.pkl",
        "type": "lstm",
        "name": "LSTM",
        "description": "Long Short-Term Memory network optimized for sequence analysis",
        "color": "#f59e0b"
    }
}

# ==================== LOAD MODEL CONFIG ====================
@st.cache_resource
def load_model_config():
    """Load model configuration from pickle file"""
    try:
        with open("model_config.pkl", "rb") as f:
            config = pickle.load(f)
        return config
    except Exception as e:
        return None

# ==================== LOAD TRANSFORMER MODEL ====================
@st.cache_resource
def load_transformer_model(model_key):
    """Load transformer model and tokenizer"""
    config = MODEL_CONFIGS[model_key]
    model_path = config["path"]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return model, tokenizer, device, None
    except Exception as e:
        return None, None, None, str(e)

# ==================== LOAD LSTM MODEL ====================
@st.cache_resource
def load_lstm_model():
    """Load LSTM model and tokenizer"""
    try:
        model = keras.models.load_model("lstm_model.keras")
        
        with open("lstm_tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)

# ==================== PREDICTION FUNCTIONS ====================
def predict_transformer(text, model, tokenizer, device):
    """Predict using transformer model"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item()
    
    label = "REAL" if prediction == 1 else "FAKE"
    
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "FAKE": probabilities[0][0].item(),
            "REAL": probabilities[0][1].item()
        }
    }

def predict_lstm(text, model, tokenizer, max_length=200):
    """Predict using LSTM model"""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    prediction = model.predict(padded, verbose=0)
    
    if prediction.shape[-1] == 1:
        prob_real = prediction[0][0]
        prob_fake = 1 - prob_real
        label = "REAL" if prob_real > 0.5 else "FAKE"
        confidence = prob_real if label == "REAL" else prob_fake
    else:
        prob_fake = prediction[0][0]
        prob_real = prediction[0][1]
        label = "REAL" if prob_real > prob_fake else "FAKE"
        confidence = max(prob_real, prob_fake)
    
    return {
        "label": label,
        "confidence": float(confidence),
        "probabilities": {
            "FAKE": float(prob_fake),
            "REAL": float(prob_real)
        }
    }

# ==================== LOAD DATASET ====================
@st.cache_data
def load_dataset():
    """Load dataset for samples"""
    try:
        df = pd.read_csv("dataset.csv")
        return df
    except:
        return None

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Fake News Detector</h1>
        <p>Advanced AI-powered news verification using state-of-the-art NLP models</p>
        <span class="header-badge">BERT • DistilBERT • LSTM</span>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Configuration</p>', unsafe_allow_html=True)
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: MODEL_CONFIGS[x]['name']
        )
        
        config = MODEL_CONFIGS[selected_model]
        badge_class = "badge-transformer" if config['type'] == 'transformer' else "badge-lstm"
        
        st.markdown(f"""
        <div class="model-selector">
            <h4>{config['name']}</h4>
            <p>{config['description']}</p>
            <span class="model-badge {badge_class}">{config['type']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Device info
        st.markdown('<p class="sidebar-title">System</p>', unsafe_allow_html=True)
        
        if config['type'] == 'transformer':
            is_gpu = torch.cuda.is_available()
        else:
            is_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        
        device_class = "" if is_gpu else "device-badge-cpu"
        device_text = "GPU Enabled" if is_gpu else "CPU Mode"
        
        st.markdown(f"""
        <div class="device-badge {device_class}">
            <span>{"◆" if is_gpu else "○"}</span>
            {device_text}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # No "Quick Actions" section anymore
        st.markdown('<p style="color: #475569; font-size: 0.75rem; text-align: center;">Built with Streamlit</p>', unsafe_allow_html=True)
    
    # ==================== MAIN CONTENT ====================
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Detector", "Model Comparison", "Dataset"])
    
    # ==================== TAB 1: DETECTOR ====================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<p class="section-title">Input News Article</p>', unsafe_allow_html=True)
            
            default_text = st.session_state.get('sample_text', '')
            
            news_text = st.text_area(
                "Paste your news article here:",
                value=default_text,
                height=220,
                placeholder="Enter the news article text you want to verify.\n\nFor best results, include the full article or at least 2-3 paragraphs with sufficient context.",
                label_visibility="collapsed"
            )
            
            if 'sample_text' in st.session_state:
                del st.session_state['sample_text']
            
            char_count = len(news_text)
            word_count = len(news_text.split()) if news_text else 0
            
            st.markdown(f"""
            <div class="stats-row">
                <span class="stat-item"><strong>{char_count:,}</strong> characters</span>
                <span class="stat-item"><strong>{word_count}</strong> words</span>
            </div>
            """, unsafe_allow_html=True)
            
            analyze_btn = st.button("Analyze Article", use_container_width=True)
        
        with col2:
            st.markdown('<p class="section-title">Guidelines</p>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="tips-card">
                <h3>For accurate results</h3>
                <div class="tip-item">
                    <div class="tip-icon">✓</div>
                    <span>Use complete sentences with proper context</span>
                </div>
                <div class="tip-item">
                    <div class="tip-icon">✓</div>
                    <span>Include at least 50 words for better analysis</span>
                </div>
                <div class="tip-item">
                    <div class="tip-icon">✓</div>
                    <span>Avoid headlines only - provide article body</span>
                </div>
                <div class="tip-item">
                    <div class="tip-icon">✓</div>
                    <span>Include factual claims for verification</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="model-selector" style="margin-top: 1rem;">
                <h4>Active Model</h4>
                <p style="color: {config['color']}; font-weight: 600;">{config['name']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis section
        if analyze_btn:
            if not news_text.strip():
                st.markdown("""
                <div class="warning-box">
                    <p>Please enter some text to analyze</p>
                </div>
                """, unsafe_allow_html=True)
            elif word_count < 5:
                st.markdown("""
                <div class="warning-box">
                    <p>Please enter at least 5 words for better accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner(f"Loading {config['name']}..."):
                    if config['type'] == 'transformer':
                        model, tokenizer, device, error = load_transformer_model(selected_model)
                    else:
                        model, tokenizer, error = load_lstm_model()
                        device = None
                
                if error:
                    st.error(f"Error loading model: {error}")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.markdown('<p style="color: #64748b; font-size: 0.875rem;">Preprocessing text...</p>', unsafe_allow_html=True)
                    progress_bar.progress(30)
                    time.sleep(0.3)
                    
                    status_text.markdown('<p style="color: #64748b; font-size: 0.875rem;">Running inference...</p>', unsafe_allow_html=True)
                    progress_bar.progress(60)
                    
                    if config['type'] == 'transformer':
                        result = predict_transformer(news_text, model, tokenizer, device)
                    else:
                        result = predict_lstm(news_text, model, tokenizer)
                    
                    status_text.markdown('<p style="color: #64748b; font-size: 0.875rem;">Generating results...</p>', unsafe_allow_html=True)
                    progress_bar.progress(100)
                    time.sleep(0.2)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # ==================== RESULTS ====================
                    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<p class="section-title">Analysis Results</p>', unsafe_allow_html=True)
                    
                    res_col1, res_col2 = st.columns([1, 1])
                    
                    with res_col1:
                        if result["label"] == "REAL":
                            st.markdown(f"""
                            <div class="result-card result-card-real">
                                <div class="result-icon result-icon-real">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="3">
                                        <polyline points="20 6 9 17 4 12"></polyline>
                                    </svg>
                                </div>
                                <p class="result-label">Verified Real</p>
                                <p class="confidence-text">Confidence: {result['confidence']*100:.1f}%</p>
                                <div class="confidence-bar">
                                    <div class="confidence-fill confidence-fill-real" style="width: {result['confidence']*100}%"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card result-card-fake">
                                <div class="result-icon result-icon-fake">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="3">
                                        <line x1="18" y1="6" x2="6" y2="18"></line>
                                        <line x1="6" y1="6" x2="18" y2="18"></line>
                                    </svg>
                                </div>
                                <p class="result-label">Likely Fake</p>
                                <p class="confidence-text">Confidence: {result['confidence']*100:.1f}%</p>
                                <div class="confidence-bar">
                                    <div class="confidence-fill confidence-fill-fake" style="width: {result['confidence']*100}%"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with res_col2:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result['probabilities']['REAL'] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Authenticity Score", 'font': {'color': '#e2e8f0', 'size': 14}},
                            number={'suffix': '%', 'font': {'color': '#e2e8f0', 'size': 36}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickcolor': '#475569', 'tickwidth': 1},
                                'bar': {'color': '#3b82f6', 'thickness': 0.8},
                                'bgcolor': 'rgba(255,255,255,0.05)',
                                'borderwidth': 0,
                                'steps': [
                                    {'range': [0, 35], 'color': 'rgba(239, 68, 68, 0.2)'},
                                    {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.2)'},
                                    {'range': [65, 100], 'color': 'rgba(34, 197, 94, 0.2)'}
                                ],
                                'threshold': {
                                    'line': {'color': '#e2e8f0', 'width': 2},
                                    'thickness': 0.8,
                                    'value': result['probabilities']['REAL'] * 100
                                }
                            }
                        ))
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': '#e2e8f0'},
                            height=220,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Probability bars
                    st.markdown("#### Probability Distribution")
                    
                    prob_df = pd.DataFrame({
                        'Category': ['Fake', 'Real'],
                        'Probability': [result['probabilities']['FAKE'] * 100, result['probabilities']['REAL'] * 100],
                    })
                    
                    fig_bar = px.bar(
                        prob_df,
                        x='Probability',
                        y='Category',
                        orientation='h',
                        color='Category',
                        color_discrete_map={'Fake': '#ef4444', 'Real': '#22c55e'},
                        text=prob_df['Probability'].apply(lambda x: f'{x:.1f}%')
                    )
                    fig_bar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#e2e8f0'},
                        showlegend=False,
                        height=120,
                        margin=dict(l=0, r=0, t=10, b=10),
                        xaxis={'showgrid': False, 'showticklabels': False, 'range': [0, 105]},
                        yaxis={'showgrid': False}
                    )
                    fig_bar.update_traces(
                        textposition='outside',
                        textfont_size=12,
                        marker_line_width=0
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <p><strong>Model:</strong> {config['name']} | <strong>Words analyzed:</strong> {word_count} | <strong>Type:</strong> {config['type'].upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ==================== TAB 2: MODEL COMPARISON ====================
    with tab2:
        st.markdown('<p class="section-title">Compare All Models</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #64748b; margin-bottom: 1.5rem;">Analyze the same text with all available models to compare their predictions.</p>', unsafe_allow_html=True)
        
        compare_text = st.text_area(
            "Enter text for comparison:",
            height=150,
            placeholder="Paste news text here to compare predictions across all models...",
            key="compare_text",
            label_visibility="collapsed"
        )
        
        if st.button("Run Comparison", use_container_width=True, key="compare_btn"):
            if not compare_text.strip():
                st.markdown("""
                <div class="warning-box">
                    <p>Please enter some text to compare</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                results_list = []
                
                progress = st.progress(0)
                status = st.empty()
                
                for idx, (model_key, model_config) in enumerate(MODEL_CONFIGS.items()):
                    status.markdown(f'<p style="color: #64748b; font-size: 0.875rem;">Loading {model_config["name"]}...</p>', unsafe_allow_html=True)
                    
                    try:
                        if model_config['type'] == 'transformer':
                            model, tokenizer, device, error = load_transformer_model(model_key)
                            if not error:
                                result = predict_transformer(compare_text, model, tokenizer, device)
                        else:
                            model, tokenizer, error = load_lstm_model()
                            if not error:
                                result = predict_lstm(compare_text, model, tokenizer)
                        
                        if not error:
                            results_list.append({
                                "Model": model_config['name'],
                                "Type": model_config['type'].upper(),
                                "Prediction": result['label'],
                                "Confidence": f"{result['confidence']*100:.1f}%",
                                "Real %": result['probabilities']['REAL'] * 100,
                                "Fake %": result['probabilities']['FAKE'] * 100
                            })
                    except Exception as e:
                        st.warning(f"Error with {model_config['name']}: {str(e)}")
                    
                    progress.progress((idx + 1) / len(MODEL_CONFIGS))
                
                progress.empty()
                status.empty()
                
                if results_list:
                    st.markdown("#### Comparison Results")
                    
                    df = pd.DataFrame(results_list)
                    st.dataframe(
                        df[["Model", "Type", "Prediction", "Confidence"]],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Real',
                        x=[r["Model"] for r in results_list],
                        y=[r["Real %"] for r in results_list],
                        marker_color='#22c55e',
                        marker_line_width=0
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Fake',
                        x=[r["Model"] for r in results_list],
                        y=[r["Fake %"] for r in results_list],
                        marker_color='#ef4444',
                        marker_line_width=0
                    ))
                    
                    fig.update_layout(
                        barmode='group',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#e2e8f0'},
                        legend={
                            'orientation': 'h',
                            'y': 1.1,
                            'x': 0.5,
                            'xanchor': 'center',
                            'font': {'size': 12}
                        },
                        height=350,
                        xaxis={'showgrid': False, 'tickfont': {'size': 12}},
                        yaxis={
                            'showgrid': True,
                            'gridcolor': 'rgba(255,255,255,0.05)',
                            'title': 'Probability %',
                            'tickfont': {'size': 11}
                        },
                        bargap=0.3,
                        bargroupgap=0.1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    predictions = [r["Prediction"] for r in results_list]
                    consensus = max(set(predictions), key=predictions.count)
                    consensus_count = predictions.count(consensus)
                    consensus_class = "consensus-real" if consensus == "REAL" else "consensus-fake"
                    
                    st.markdown(f"""
                    <div class="consensus-box">
                        <h3>Model Consensus</h3>
                        <p class="consensus-result {consensus_class}">{consensus_count}/{len(predictions)} models predict: {consensus}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ==================== TAB 3: DATASET INFO ====================
    with tab3:
        st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)
        
        df = load_dataset()
        
        if df is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h2 style="color: #3b82f6;">{len(df):,}</h2>
                    <p>Total Samples</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                label_col = None
                for col in ['label', 'Label', 'LABEL', 'class', 'target']:
                    if col in df.columns:
                        label_col = col
                        break
                
                if label_col:
                    real_count = len(df[df[label_col].isin([1, 'REAL', 'real', 'Real', 'TRUE', 'true'])])
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <h2 style="color: #22c55e;">{real_count:,}</h2>
                        <p>Real News</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if label_col:
                    fake_count = len(df[df[label_col].isin([0, 'FAKE', 'fake', 'Fake', 'FALSE', 'false'])])
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <h2 style="color: #ef4444;">{fake_count:,}</h2>
                        <p>Fake News</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if label_col:
                st.markdown("#### Label Distribution")
                
                label_counts = df[label_col].value_counts()
                
                fig = px.pie(
                    values=label_counts.values,
                    names=['Fake', 'Real'] if label_counts.index[0] in [0, 'FAKE', 'fake', 'Fake'] else ['Real', 'Fake'],
                    color_discrete_sequence=['#ef4444', '#22c55e'],
                    hole=0.5
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'},
                    height=300,
                    showlegend=True,
                    legend={
                        'orientation': 'h',
                        'y': -0.1,
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent',
                    textfont_size=14
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Data Preview")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            
            st.markdown("#### Dataset Columns")
            cols_text = ", ".join(list(df.columns))
            st.markdown(f"""
            <div class="info-box">
                <p>{cols_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="warning-box">
                <p>Dataset file (dataset.csv) not found. Please ensure the file is in the correct location.</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()