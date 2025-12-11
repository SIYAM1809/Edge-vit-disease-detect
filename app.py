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
# 1. CONFIGURATION & GREEN GRADIENT THEME
# ==============================================================================
st.set_page_config(
    page_title="Edge-ViT Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Green Gradient + Multi-Page Styling
st.markdown("""
    <style>
    /* 1. MAIN BACKGROUND: Green-White Gradient */
    .stApp {
        background: linear-gradient(to bottom, #E8F5E9, #FFFFFF);
        background-attachment: fixed;
    }
    
    /* 2. TEXT VISIBILITY: Force Dark Green/Charcoal */
    h1, h2, h3, h4, h5, h6 {
        color: #1B5E20 !important; /* Dark Forest Green */
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
    }
    
    p, label, .stMarkdown, li, .stCaption, .stText {
        color: #2E3B33 !important; /* Dark Charcoal */
        font-size: 1.1rem !important;
        font-weight: 500;
    }

    /* 3. CARDS (Glassmorphism Effect) */
    .css-1r6slb0, .stFileUploader, .element-container { 
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 10px;
    }
    
    /* 4. AGILE BUTTON STYLE */
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
    }
    
    /* 5. Custom Success/Error Boxes */
    div.stSuccess {
        border-left: 6px solid #2E7D32;
        background-color: #F1F8E9;
    }
    div.stError {
        border-left: 6px solid #C62828;
        background-color: #FFEBEE;
    }
    div.stWarning {
        border-left: 6px solid #EF6C00;
        background-color: #FFF3E0;
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
            st.error(f"Failed to load weights: {e}")
    else:
        st.error(f"CRITICAL ERROR: Model file not found!")
    
    model.to(device)
    model.eval()
    return model

model = load_model()

# ==============================================================================
# 3. SESSION STATE MANAGEMENT (MULTI-PAGE LOGIC)
# ==============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

def navigate_to(page):
    st.session_state.page = page

# ==============================================================================
# 4. PAGE 1: HOME (UPLOAD)
# ==============================================================================
def render_home():
    st.title("🌿 Intelligent Crop Disease Diagnosis")
    st.markdown("#### Precision Agriculture using Saliency-Guided AI")
    st.write("Upload a specimen below. The system will analyze pathology using Edge-ViT.")

    # Centered Input Card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div style="background-color: rgba(255,255,255,0.8); padding: 30px; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Specimen")
        uploaded_file = st.file_uploader("Choose File", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.uploaded_image = image
            
            # Preview
            st.markdown(
                f'<div style="border-radius: 12px; overflow: hidden; border: 2px solid #C8E6C9; margin-top: 10px;">',
                unsafe_allow_html=True
            )
            st.image(image, caption='Specimen Preview', use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 RUN DIAGNOSIS"):
                navigate_to('result')
                st.rerun() # Force reload to show result page
                
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 5. PAGE 2: RESULT REPORT
# ==============================================================================
def render_result():
    st.title("📊 Diagnostic Report")
    
    # Back Button (Top Left)
    if st.button("⬅️ Analyze Another Sample"):
        navigate_to('home')
        st.session_state.uploaded_image = None
        st.rerun()

    image = st.session_state.uploaded_image
    
    if image is not None:
        # Layout: Side by Side
        col_viz, col_data = st.columns([1, 1], gap="large")
        
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

        # Heatmap Gen
        heatmap = cv2.resize(attn_maps[0, 0].numpy(), (224, 224))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        if is_healthy:
            status_color = "#2E7D32" # Dark Green
            bg_color = "#E8F5E9"
            icon = "✅"
            title = "Healthy Specimen"
            msg = f"High confidence ({confidence_score*100:.1f}%) of plant health."
            overlay = np.array(image.resize((224, 224))) / 255.0
            caption = "Visualization: Clear Leaf Surface"
        elif is_unknown:
            status_color = "#C62828" # Red
            bg_color = "#FFEBEE"
            icon = "❓"
            title = "Inconclusive Result"
            msg = "Confidence below 50%. This may be a non-plant image."
            overlay = np.array(image.resize((224, 224))) / 255.0
            caption = "Visualization: Suppressed"
        else:
            status_color = "#EF6C00" # Orange
            bg_color = "#FFF3E0"
            icon = "⚠️"
            title = f"{readable_name}"
            msg = f"Pathology detected with **{confidence_score*100:.1f}% confidence**."
            
            heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB) / 255.0
            original_np = np.array(image.resize((224, 224))) / 255.0
            overlay = 0.6 * original_np + 0.4 * heatmap_c
            caption = "Saliency Map: Red Indicates Infection"

        # --- RIGHT COLUMN: DETAILS ---
        with col_data:
            st.markdown(f"""
            <div style="background-color: {bg_color}; border-left: 8px solid {status_color}; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <h2 style="color: {status_color} !important; margin: 0;">{icon} {title}</h2>
                <p style="font-size: 1.2rem; margin-top: 10px; color: #333 !important;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📋 Recommendations")
            if is_healthy:
                st.success("No action needed. Continue standard care.")
            elif is_unknown:
                st.error("Please re-upload a clearer image of a single leaf.")
            else:
                st.warning(f"Isolate the affected plant immediately to prevent spread of **{readable_name.split('-')[-1]}**.")
                
            with st.expander("🔬 View Latent Features"):
                st.json({
                    "Class ID": pred_idx.item(),
                    "Backbone": "MobileViT-XS",
                    "Attention Score": f"{heatmap.mean():.4f}"
                })

        # --- LEFT COLUMN: VISUALS ---
        with col_viz:
            st.markdown(f'<div style="background-color: white; padding: 15px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
            st.markdown(f"**{caption}**")
            st.image(overlay, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 6. APP ROUTER
# ==============================================================================
if st.session_state.page == 'home':
    render_home()
elif st.session_state.page == 'result':
    render_result()