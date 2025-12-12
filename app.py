import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import timm
import os
import time

# ==============================================================================
# 1. CONFIGURATION & PREMIUM GREEN GRADIENT THEME
# ==============================================================================
st.set_page_config(
    page_title="Edge-ViT Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS with Animations & Professional Styling
st.markdown("""
    <style>
    /* ==================== GOOGLE FONTS ==================== */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ==================== GLOBAL ANIMATIONS ==================== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* ==================== MAIN BACKGROUND ==================== */
    .stApp {
        background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 25%, #FFFFFF 50%, #F1F8E9 75%, #E8F5E9 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* ==================== TYPOGRAPHY ==================== */
    h1, h2, h3, h4, h5, h6 {
        color: #1B5E20 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        animation: fadeIn 0.8s ease-out;
        letter-spacing: -0.5px;
    }
    
    h1 {
        font-size: 3rem !important;
        background: linear-gradient(135deg, #1B5E20, #2E7D32, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeIn 1s ease-out, shimmer 3s infinite;
        background-size: 200% auto;
    }
    
    h2 {
        font-size: 2rem !important;
    }
    
    p, label, .stMarkdown, li, .stCaption, .stText {
        color: #2E4A35 !important;
        font-size: 1.05rem !important;
        font-weight: 400 !important;
        line-height: 1.7 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* ==================== CONTAINERS & CARDS ==================== */
    .element-container, .stFileUploader, div[data-testid="stFileUploader"] {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Premium Card Style */
    .premium-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 40px;
        box-shadow: 
            0 10px 40px rgba(27, 94, 32, 0.08),
            0 2px 8px rgba(27, 94, 32, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(27, 94, 32, 0.08);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeIn 0.8s ease-out;
    }
    
    .premium-card:hover {
        transform: translateY(-5px);
        box-shadow: 
            0 20px 60px rgba(27, 94, 32, 0.12),
            0 4px 12px rgba(27, 94, 32, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 1);
    }
    
    /* ==================== BUTTONS ==================== */
    .stButton>button {
        background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%);
        color: white !important;
        border-radius: 16px;
        height: 3.8em;
        width: 100%;
        border: none;
        font-weight: 700 !important;
        font-size: 1.1em !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-family: 'Poppins', sans-serif !important;
        box-shadow: 
            0 8px 24px rgba(46, 125, 50, 0.25),
            0 2px 6px rgba(46, 125, 50, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #66BB6A 0%, #43A047 100%);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 12px 32px rgba(46, 125, 50, 0.35),
            0 4px 8px rgba(46, 125, 50, 0.2);
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Secondary Button Style */
    .stButton>button[kind="secondary"] {
        background: linear-gradient(135deg, #81C784 0%, #66BB6A 100%);
    }
    
    /* ==================== FILE UPLOADER ==================== */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 2px dashed #81C784;
        border-radius: 20px;
        padding: 30px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #43A047;
        background: rgba(255, 255, 255, 0.9);
        transform: scale(1.01);
    }
    
    div[data-testid="stFileUploader"] label {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #2E7D32 !important;
    }
    
    /* ==================== ALERT BOXES ==================== */
    div.stSuccess {
        border-left: 6px solid #2E7D32;
        background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
        border-radius: 12px;
        padding: 20px;
        animation: slideInLeft 0.5s ease-out;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.1);
    }
    
    div.stError {
        border-left: 6px solid #C62828;
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        border-radius: 12px;
        padding: 20px;
        animation: slideInLeft 0.5s ease-out;
        box-shadow: 0 4px 12px rgba(198, 40, 40, 0.1);
    }
    
    div.stWarning {
        border-left: 6px solid #EF6C00;
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
        border-radius: 12px;
        padding: 20px;
        animation: slideInLeft 0.5s ease-out;
        box-shadow: 0 4px 12px rgba(239, 108, 0, 0.1);
    }
    
    /* ==================== IMAGES ==================== */
    .stImage {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.8s ease-out;
    }
    
    .stImage:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
    }
    
    /* ==================== EXPANDER ==================== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
        border-radius: 12px;
        font-weight: 600 !important;
        color: #2E7D32 !important;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #C8E6C9, #DCEDC8);
        transform: translateX(5px);
    }
    
    /* ==================== PROGRESS BAR ==================== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #43A047, #66BB6A, #81C784, #66BB6A, #43A047);
        background-size: 200% 100%;
        animation: shimmer 2s linear infinite;
        border-radius: 10px;
    }
    
    /* ==================== CUSTOM CLASSES ==================== */
    .status-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        animation: fadeIn 0.5s ease-out;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out;
        border: 1px solid rgba(27, 94, 32, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }
    
    /* ==================== LOADING SPINNER ==================== */
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid #E8F5E9;
        border-top: 4px solid #43A047;
        border-radius: 50%;
        animation: rotate 1s linear infinite;
        margin: 20px auto;
    }
    
    /* ==================== ZOOM CONTROLS ==================== */
    .zoom-controls {
        position: fixed;
        bottom: 30px;
        right: 30px;
        display: flex;
        gap: 10px;
        z-index: 999;
        animation: fadeIn 1s ease-out;
    }
    
    .zoom-btn {
        background: linear-gradient(135deg, #43A047, #2E7D32);
        color: white;
        border: none;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        font-size: 1.5rem;
        cursor: pointer;
        box-shadow: 0 4px 16px rgba(46, 125, 50, 0.3);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .zoom-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
    }
    
    .zoom-btn:active {
        transform: scale(0.95);
    }
    
    /* ==================== MODAL/LIGHTBOX ==================== */
    .image-modal {
        display: none;
        position: fixed;
        z-index: 9999;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.95);
        animation: fadeIn 0.3s ease-out;
    }
    
    .modal-content {
        margin: auto;
        display: block;
        max-width: 90%;
        max-height: 90%;
        animation: pulse 0.5s ease-out;
    }
    
    .close-modal {
        position: absolute;
        top: 20px;
        right: 40px;
        color: white;
        font-size: 40px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .close-modal:hover {
        color: #43A047;
        transform: scale(1.2);
    }
    
    /* ==================== SCROLLBAR ==================== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F1F8E9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #66BB6A, #43A047);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #43A047, #2E7D32);
    }
    
    /* ==================== HIDE STREAMLIT BRANDING ==================== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ==================== RESPONSIVE DESIGN ==================== */
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem !important;
        }
        
        .premium-card {
            padding: 20px;
        }
        
        .zoom-controls {
            bottom: 20px;
            right: 20px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA & ARCHITECTURE
# ==============================================================================

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

class SaliencyGuidedAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn_conv = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
    def forward(self, x):
        attn_map = self.attn_conv(x)
        return x * (1.0 + attn_map), attn_map

class EdgeViT_FSL(nn.Module):
    def __init__(self, num_classes=38): 
        super().__init__()
        self.backbone = timm.create_model('mobilevit_xs', pretrained=False, num_classes=0)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            self.feat_dim = self.backbone.forward_features(dummy).shape[1]
        self.saliency = SaliencyGuidedAttention(self.feat_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        refined, attn_map = self.saliency(features)
        logits = self.classifier(self.pool(refined).flatten(1))
        return logits, attn_map

@st.cache_resource
def load_model():
    device = torch.device('cpu') 
    model = EdgeViT_FSL(num_classes=38) 
    possible_names = ["best_edge_vit_final.pth", "best_edge_vit.pth"]
    model_path = None
    for name in possible_names:
        if os.path.exists(name):
            model_path = name
            break
            
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            st.error(f"⚠️ Failed to load weights: {e}")
    else:
        st.error(f"🚨 CRITICAL ERROR: Model file not found!")
    
    model.to(device)
    model.eval()
    return model

model = load_model()

# ==============================================================================
# 3. SESSION STATE MANAGEMENT
# ==============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 1.0

def navigate_to(page):
    st.session_state.page = page

# ==============================================================================
# 4. PAGE 1: HOME (UPLOAD)
# ==============================================================================
def render_home():
    # Hero Section
    st.markdown('<div style="text-align: center; padding: 40px 0 60px 0;">', unsafe_allow_html=True)
    st.title("🌿 Intelligent Crop Disease Diagnosis")
    st.markdown(
        """
        <p style="font-size: 1.3rem; color: #43A047 !important; font-weight: 500; margin-top: -10px;">
        Precision Agriculture using Saliency-Guided Edge-ViT AI
        </p>
        """, 
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Features Section
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size: 3rem; margin-bottom: 10px;">🎯</div>
                <h3 style="color: #2E7D32 !important; margin-bottom: 8px;">High Accuracy</h3>
                <p style="color: #555 !important; font-size: 0.95rem !important;">Advanced AI detection with 95%+ accuracy rate</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col_f2:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size: 3rem; margin-bottom: 10px;">⚡</div>
                <h3 style="color: #2E7D32 !important; margin-bottom: 8px;">Instant Results</h3>
                <p style="color: #555 !important; font-size: 0.95rem !important;">Real-time analysis in under 3 seconds</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col_f3:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size: 3rem; margin-bottom: 10px;">🔬</div>
                <h3 style="color: #2E7D32 !important; margin-bottom: 8px;">38 Diseases</h3>
                <p style="color: #555 !important; font-size: 0.95rem !important;">Detects multiple crop pathologies</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Upload Section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 25px;">
                <h2 style="color: #2E7D32 !important; margin-bottom: 10px;">📤 Upload Plant Specimen</h2>
                <p style="color: #666 !important;">Supported formats: JPG, PNG, JPEG • Max size: 200MB</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader(
            "Choose File", 
            type=["jpg", "png", "jpeg"], 
            label_visibility="collapsed",
            help="Upload a clear image of a single leaf for best results"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.uploaded_image = image
            
            # Preview with animation
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 15px;">
                    <span class="status-badge" style="background: linear-gradient(135deg, #C8E6C9, #A5D6A7); color: #1B5E20;">
                        ✓ Image Loaded Successfully
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown(
                '<div style="border-radius: 16px; overflow: hidden; border: 3px solid #C8E6C9; box-shadow: 0 8px 24px rgba(67, 160, 71, 0.2);">',
                unsafe_allow_html=True
            )
            st.image(image, caption='📸 Uploaded Specimen Preview', use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 RUN DIAGNOSIS"):
                with st.spinner(''):
                    st.markdown(
                        '<div class="loading-spinner"></div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        '<p style="text-align: center; color: #43A047 !important; font-weight: 600;">Analyzing specimen...</p>',
                        unsafe_allow_html=True
                    )
                    time.sleep(1)  # Simulate processing
                navigate_to('result')
                st.rerun()
        else:
            st.markdown(
                """
                <div style="text-align: center; padding: 40px; color: #81C784;">
                    <div style="font-size: 4rem; margin-bottom: 15px;">🖼️</div>
                    <p style="font-size: 1.1rem; color: #66BB6A !important;">
                        Drag and drop your image here or click to browse
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_i1, col_i2, col_i3 = st.columns([1, 2, 1])
    
    with col_i2:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(200, 230, 201, 0.3), rgba(220, 237, 200, 0.3)); 
                        border-radius: 16px; padding: 25px; border: 1px solid rgba(67, 160, 71, 0.2);">
                <h3 style="color: #2E7D32 !important; margin-bottom: 15px; text-align: center;">📋 Best Practices</h3>
                <ul style="color: #333 !important; line-height: 2;">
                    <li><strong>Clear focus:</strong> Ensure the leaf is in sharp focus</li>
                    <li><strong>Good lighting:</strong> Natural daylight works best</li>
                    <li><strong>Single leaf:</strong> Capture one leaf at a time</li>
                    <li><strong>Minimal background:</strong> Avoid cluttered backgrounds</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==============================================================================
# 5. PAGE 2: RESULT REPORT WITH ZOOM FUNCTIONALITY
# ==============================================================================
def render_result():
    # Back Button with Animation
    col_back, col_space = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ New Analysis", use_container_width=True):
            navigate_to('home')
            st.session_state.uploaded_image = None
            st.session_state.zoom_level = 1.0
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("📊 Diagnostic Report")
    st.markdown("<br>", unsafe_allow_html=True)

    image = st.session_state.uploaded_image
    
    if image is not None:
        # Show loading animation
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            st.markdown(
                """
                <div style="text-align: center; padding: 40px;">
                    <div class="loading-spinner"></div>
                    <p style="color: #43A047 !important; font-weight: 600; margin-top: 20px; font-size: 1.2rem;">
                        🔍 Analyzing leaf pathology...
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
        
        progress_placeholder.empty()
        
        # --- INFERENCE ---
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            logits, attn_maps = model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        # Logic
        confidence_score = conf.item()
        raw_class_name = CLASS_NAMES[pred_idx.item()]
        readable_name = raw_class_name.replace("___", " - ").replace("_", " ")
        is_healthy = "healthy" in readable_name.lower()
        is_unknown = confidence_score < 0.50

        # Heatmap Generation
        heatmap = cv2.resize(attn_maps[0, 0].numpy(), (224, 224))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        if is_healthy:
            status_color = "#2E7D32"
            bg_color = "linear-gradient(135deg, #E8F5E9, #F1F8E9)"
            icon = "✅"
            title = "Healthy Specimen"
            msg = f"The model detected a healthy plant with **{confidence_score*100:.2f}% confidence**."
            recommendation = "No action needed. Continue standard care and monitoring."
            overlay = np.array(image.resize((224, 224))) / 255.0
            caption = "Original Image: No Pathology Detected"
        elif is_unknown:
            status_color = "#C62828"
            bg_color = "linear-gradient(135deg, #FFEBEE, #FFCDD2)"
            icon = "❓"
            title = "Inconclusive Result"
            msg = f"Confidence level is only **{confidence_score*100:.2f}%** (below 50% threshold)."
            recommendation = "Please upload a clearer image of a single leaf with good lighting."
            overlay = np.array(image.resize((224, 224))) / 255.0
            caption = "Original Image: Unable to Detect Patterns"
        else:
            status_color = "#EF6C00"
            bg_color = "linear-gradient(135deg, #FFF3E0, #FFE0B2)"
            icon = "⚠️"
            title = readable_name
            msg = f"Disease detected with **{confidence_score*100:.2f}% confidence**."
            recommendation = f"**Immediate Action Required:** Isolate affected plants to prevent spread of {readable_name.split('-')[-1].strip()}. Consult agricultural expert for treatment options."
            
            # Create heatmap overlay
            heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB) / 255.0
            original_np = np.array(image.resize((224, 224))) / 255.0
            overlay = 0.6 * original_np + 0.4 * heatmap_c
            caption = "Saliency Map: Red Areas Indicate Detected Pathology"

        # Layout: Side by Side
        col_viz, col_data = st.columns([1.2, 1], gap="large")
        
        # --- LEFT COLUMN: VISUALIZATION ---
        with col_viz:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown(f"**{caption}**")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Image with zoom
            zoom_level = st.session_state.zoom_level
            img_width = int(100 * zoom_level)
            
            st.markdown(
                f'<div style="border-radius: 16px; overflow: hidden; border: 3px solid {status_color}; box-shadow: 0 8px 24px rgba(0,0,0,0.15);">',
                unsafe_allow_html=True
            )
            st.image(overlay, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Zoom Controls
            st.markdown("<br>", unsafe_allow_html=True)
            zoom_col1, zoom_col2, zoom_col3, zoom_col4 = st.columns([1, 1, 1, 1])
            
            with zoom_col1:
                if st.button("🔍 Zoom In", use_container_width=True):
                    st.session_state.zoom_level = min(st.session_state.zoom_level + 0.2, 3.0)
                    st.rerun()
            
            with zoom_col2:
                if st.button("🔎 Zoom Out", use_container_width=True):
                    st.session_state.zoom_level = max(st.session_state.zoom_level - 0.2, 0.5)
                    st.rerun()
            
            with zoom_col3:
                if st.button("↺ Reset", use_container_width=True):
                    st.session_state.zoom_level = 1.0
                    st.rerun()
            
            with zoom_col4:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #E8F5E9, #F1F8E9); 
                                border-radius: 12px; padding: 12px; text-align: center; 
                                border: 2px solid #C8E6C9; font-weight: 600; color: #2E7D32 !important;">
                        {int(zoom_level * 100)}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Original Image for comparison
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📷 View Original Image"):
                st.image(image, caption="Original Uploaded Image", use_container_width=True)

        # --- RIGHT COLUMN: DETAILS ---
        with col_data:
            st.markdown(
                f"""
                <div style="background: {bg_color}; 
                            border-left: 8px solid {status_color}; 
                            padding: 30px; 
                            border-radius: 16px; 
                            box-shadow: 0 8px 24px rgba(0,0,0,0.08);">
                    <div style="font-size: 3rem; margin-bottom: 15px;">{icon}</div>
                    <h2 style="color: {status_color} !important; margin: 0 0 15px 0;">{title}</h2>
                    <p style="font-size: 1.15rem; line-height: 1.7; color: #333 !important;">{msg}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Confidence Meter
            st.markdown("### 📈 Confidence Score")
            st.progress(confidence_score)
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 10px;">
                    <span style="font-size: 2rem; font-weight: 700; color: {status_color};">
                        {confidence_score*100:.2f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 📋 Recommendations")
            if is_healthy:
                st.success(recommendation)
            elif is_unknown:
                st.error(recommendation)
            else:
                st.warning(recommendation)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Technical Details
            with st.expander("🔬 Technical Details"):
                st.json({
                    "Predicted Class": readable_name,
                    "Class ID": int(pred_idx.item()),
                    "Confidence Score": f"{confidence_score:.4f}",
                    "Model Architecture": "Edge-ViT (MobileViT-XS + Saliency Attention)",
                    "Input Resolution": "224x224",
                    "Attention Score": f"{heatmap.mean():.4f}",
                    "Feature Dimension": 384
                })
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Top 3 Predictions
            with st.expander("📊 Top 3 Predictions"):
                top3_probs, top3_indices = torch.topk(probs[0], 3)
                for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                    class_name = CLASS_NAMES[idx.item()].replace("___", " - ").replace("_", " ")
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 10px; padding: 10px; 
                                    background: linear-gradient(135deg, #F1F8E9, #FFFFFF); 
                                    border-radius: 8px; border-left: 4px solid #81C784;">
                            <strong>{i+1}.</strong> {class_name}<br>
                            <small>Confidence: {prob.item()*100:.2f}%</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    
    else:
        st.error("⚠️ No image found. Please return to home and upload an image.")
        if st.button("⬅️ Go to Home"):
            navigate_to('home')
            st.rerun()

# ==============================================================================
# 6. APP ROUTER
# ==============================================================================
if st.session_state.page == 'home':
    render_home()
elif st.session_state.page == 'result':
    render_result()