import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import timm
import os

# ==============================================================================
# 1. CONFIGURATION & PROFESSIONAL GRADIENT THEME
# ==============================================================================
st.set_page_config(
    page_title="Edge-ViT Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Gradient Theme + High Contrast Text
st.markdown("""
    <style>
    /* 1. MAIN BACKGROUND: Blue-White Gradient */
    .stApp {
        background: linear-gradient(to bottom, #E3F2FD, #FFFFFF);
        background-attachment: fixed;
    }
    
    /* 2. TEXT VISIBILITY FIXED: Force Dark Colors */
    h1, h2, h3, h4, h5, h6 {
        color: #0D47A1 !important; /* Professional Navy Blue Headers */
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.5); /* Slight glow for readability */
    }
    
    p, label, .stMarkdown, li, .stCaption {
        color: #263238 !important; /* Dark Charcoal for reading */
        font-size: 1.05rem !important;
        font-weight: 500;
    }

    /* 3. CARDS (Glassmorphism Effect) */
    /* Used for Image Upload & Result Areas to make them pop */
    .css-1r6slb0, .css-12oz5g7 { 
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* 4. AGILE BUTTON STYLE (Vibrant Green) */
    .stButton>button {
        background: linear-gradient(45deg, #43A047, #2E7D32); 
        color: white !important;
        border-radius: 10px;
        height: 3.5em;
        width: 100%;
        border: none;
        font-weight: 800;
        font-size: 1.2em;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 5px 15px rgba(46, 125, 50, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #66BB6A, #43A047);
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.4);
        color: white !important;
    }

    /* 5. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFB; /* Very Light Grey/Blue */
        border-right: 1px solid #CFD8DC;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li {
        color: #37474F !important;
        font-size: 0.95rem !important;
    }
    
    /* 6. UPLOAD BOX STYLING */
    [data-testid='stFileUploader'] {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px dashed #90CAF9;
        border-radius: 12px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA & ARCHITECTURE (UNCHANGED)
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
            st.error(f"Failed to load weights: {e}")
    else:
        st.error(f"CRITICAL ERROR: Model file not found!")
    
    model.to(device)
    model.eval()
    return model

model = load_model()

# ==============================================================================
# 3. MAIN APPLICATION UI
# ==============================================================================

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microscope.png", width=70)
    st.markdown("## Edge-ViT System")
    st.markdown("---")
    st.success("**System Status: Online** 🟢")
    st.markdown("""
    ### **User Guide**
    1.  **Upload** a clear leaf image.
    2.  Click **'RUN DIAGNOSIS'**.
    3.  Review the **AI Analysis**.
    """)
    st.markdown("---")
    st.caption("v1.3.0 Gradient Theme | PyTorch")

# --- Main Content ---
st.title("🌿 Intelligent Crop Disease Diagnosis")
st.markdown("#### Precision Agriculture using Saliency-Guided AI")
st.write("This professional tool uses computer vision to detect plant pathology. Upload a sample to begin analysis.")

# Layout: Two columns (Upload Left, Result Right)
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    # Creating a visible card for input
    st.markdown('<div style="background-color: rgba(255,255,255,0.7); padding: 20px; border-radius: 15px; border: 1px solid #E3F2FD;">', unsafe_allow_html=True)
    st.markdown("### 1. Specimen Input")
    st.write("Upload a JPG/PNG image of the affected leaf.")
    uploaded_file = st.file_uploader("Choose File", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        # Professional Image Frame
        st.markdown(
            f'<div style="border-radius: 12px; overflow: hidden; border: 4px solid #FFFFFF; box-shadow: 0 8px 16px rgba(0,0,0,0.1);">',
            unsafe_allow_html=True
        )
        st.image(image, caption='Source Specimen', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 RUN DIAGNOSIS", type="primary")
    st.markdown('</div>', unsafe_allow_html=True) # End Input Card

# --- Inference Logic ---
if uploaded_file and run_btn:
    with col_result:
        # Creating a visible card for results
        st.markdown('<div style="background-color: rgba(255,255,255,0.7); padding: 20px; border-radius: 15px; border: 1px solid #E3F2FD;">', unsafe_allow_html=True)
        st.markdown("### 2. Analysis Report")
        
        with st.spinner("🧠 AI Model is processing..."):
            with torch.no_grad():
                logits, attn_maps = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, 1)
            
            # --- Logic ---
            confidence_score = conf.item()
            raw_class_name = CLASS_NAMES[pred_idx.item()]
            readable_name = raw_class_name.replace("___", " - ").replace("_", " ")
            
            is_healthy = "healthy" in readable_name.lower()
            is_unknown = confidence_score < 0.50
            
            heatmap = cv2.resize(attn_maps[0, 0].numpy(), (224, 224))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            if is_healthy:
                status_color = "#2E7D32" # Dark Green
                bg_color = "#E8F5E9"
                icon = "✅"
                title = "Healthy Specimen"
                msg = f"Confidence: **{confidence_score*100:.1f}%**"
                overlay = np.array(image.resize((224, 224))) / 255.0
                caption = "Visualization: Clear Leaf Surface"
                
            elif is_unknown:
                status_color = "#C62828" # Red
                bg_color = "#FFEBEE"
                icon = "❓"
                title = "Unknown / Anomaly"
                msg = "Confidence below 50%. Inconclusive result."
                overlay = np.array(image.resize((224, 224))) / 255.0
                caption = "Visualization: Suppressed"
                
            else:
                status_color = "#EF6C00" # Orange
                bg_color = "#FFF3E0"
                icon = "⚠️"
                title = f"{readable_name}"
                msg = f"Confidence: **{confidence_score*100:.1f}%**"
                
                # Heatmap
                heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB) / 255.0
                original_np = np.array(image.resize((224, 224))) / 255.0
                overlay = 0.6 * original_np + 0.4 * heatmap_c
                caption = "Saliency Map: Red Indicates Pathology"

            # --- Custom Result Box ---
            st.markdown(f"""
            <div style="background-color: {bg_color}; border-left: 6px solid {status_color}; padding: 15px; border-radius: 5px;">
                <h3 style="color: {status_color} !important; margin: 0;">{icon} {title}</h3>
                <p style="margin: 5px 0 0 0; color: #333 !important;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- Display Visual ---
            st.markdown(
                f'<div style="border-radius: 12px; overflow: hidden; border: 3px solid {status_color}; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">',
                unsafe_allow_html=True
            )
            st.image(overlay, caption=caption, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True) # End Result Card