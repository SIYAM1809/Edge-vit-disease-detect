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
# 1. CONFIGURATION & PROFESSIONAL THEME
# ==============================================================================
st.set_page_config(
    page_title="Edge-ViT Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Green/White Theme with High Contrast
st.markdown("""
    <style>
    /* Main Background & Default Text */
    .stApp {
        background-color: #FFFFFF;
        color: #424242; /* Professional dark charcoal for high readability */
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    /* Headers - Dark Forest Green */
    h1, h2, h3, h4, h5, h6 {
        color: #1B5E20 !important;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    /* Standard Paragraphs & Labels */
    p, label, .stMarkdown, .stSelectbox label {
        color: #424242 !important;
        line-height: 1.6;
    }
    /* Captions and smaller details */
    .stCaption {
        color: #616161 !important;
        font-size: 0.9em;
    }
    /* Primary Button - Agricultural Green */
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border-radius: 8px;
        height: 3.2em;
        width: 100%;
        border: none;
        font-weight: 700;
        font-size: 1.1em;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #1B5E20;
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
        transform: translateY(-1px);
    }
    /* Info/Success/Warning Boxes - Themed Text */
    .stSuccess, .stInfo {
        background-color: #E8F5E9;
        border-left: 6px solid #2E7D32;
        color: #1B5E20;
    }
    .stWarning {
        background-color: #FFF3E0;
        border-left: 6px solid #FF9800;
        color: #BF360C;
    }
    .stError {
        background-color: #FFEBEE;
        border-left: 6px solid #D32F2F;
        color: #B71C1C;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F1F8E9;
        border-right: 1px solid #C8E6C9;
    }
    section[data-testid="stSidebar"] h1 {
         color: #2E7D32 !important;
    }
    /* Make sidebar text sharper green */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] .stMarkdown {
         color: #33691E !important;
    }
    /* File Uploader Area */
    [data-testid='stFileUploader'] {
        border: 2px dashed #A5D6A7;
        background-color: #FAFAFA;
        padding: 20px;
        border-radius: 10px;
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

# ==============================================================================
# 3. MODEL LOADING
# ==============================================================================
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
# 4. MAIN APPLICATION UI
# ==============================================================================

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
    st.title("Edge-ViT System")
    st.markdown("---")
    st.info("**Status:** 🟢 System Online")
    st.markdown("""
    ### **How to use:**
    1.  **Upload** a clear image of a crop leaf.
    2.  Click the **'Diagnose'** button.
    3.  View the **prediction** and AI **heatmap**.
    """)
    st.markdown("---")
    st.caption("v1.1.0 Professional | Powered by MobileViT & PyTorch")

# --- Main Content ---
st.title("🌿 Intelligent Crop Disease Diagnosis")
st.markdown("#### Rapid Identification using Saliency-Guided Few-Shot Learning")
st.write("This professional system uses advanced AI to identify plant diseases and visually highlight infected regions for precise diagnostics.")

# Layout: Two columns (Upload Left, Result Right)
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("### 1. Image Input")
    st.write("Please upload a high-quality JPG or PNG image of a single leaf.")
    uploaded_file = st.file_uploader("Choose File", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        # Rounded corners with subtle border for image
        st.markdown(
            f'<div style="border-radius: 12px; overflow: hidden; border: 3px solid #C8E6C9; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">',
            unsafe_allow_html=True
        )
        st.image(image, caption='Source Image Preview', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        run_btn = st.button("🔍 Run Professional Diagnosis", type="primary")

# --- Inference Logic ---
if uploaded_file and run_btn:
    with col_result:
        st.markdown("### 2. Diagnostic Report")
        
        with st.spinner("🧠 AI Model is analyzing leaf pathology..."):
            with torch.no_grad():
                logits, attn_maps = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, 1)
            
            # --- Logic ---
            confidence_score = conf.item()
            raw_class_name = CLASS_NAMES[pred_idx.item()]
            readable_name = raw_class_name.replace("___", " - ").replace("_", " ")
            
            # --- SMART VISUALIZATION TOGGLE ---
            is_healthy = "healthy" in readable_name.lower()
            is_unknown = confidence_score < 0.50
            
            heatmap = cv2.resize(attn_maps[0, 0].numpy(), (224, 224))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            if is_healthy:
                status_color = "success"
                result_title = "Healthy Plant"
                msg = f"The assessment indicates a **healthy** plant with a high confidence of **{confidence_score*100:.1f}%**."
                overlay = np.array(image.resize((224, 224))) / 255.0
                caption = "Visualization: Clean Leaf (No pathology detected)"
                
            elif is_unknown:
                status_color = "error"
                result_title = "Unknown / Anomaly"
                msg = "Confidence is below the threshold for a reliable diagnosis. This may be a non-plant image or an unknown condition."
                overlay = np.array(image.resize((224, 224))) / 255.0
                caption = "Visualization: Heatmap suppressed due to low confidence."
                
            else:
                status_color = "warning"
                result_title = f"Detected: {readable_name}"
                msg = f"The model has identified signs of **{readable_name}** with **{confidence_score*100:.1f}% confidence**."
                
                # Create Heatmap Overlay
                heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB) / 255.0
                original_np = np.array(image.resize((224, 224))) / 255.0
                overlay = 0.6 * original_np + 0.4 * heatmap_c
                caption = "Visualization: Saliency Heatmap (Red indicates primary infection sites)"

            # --- Display Result Box ---
            if status_color == "success":
                st.success(f"## ✅ {result_title}\n\n{msg}")
            elif status_color == "warning":
                st.warning(f"## ⚠️ {result_title}\n\n{msg}")
            else:
                st.error(f"## ❓ {result_title}\n\n{msg}")
            
            # --- Display Visual ---
            st.markdown(
                f'<div style="border-radius: 12px; overflow: hidden; border: 3px solid #{ "C8E6C9" if status_color=="success" else "FFE0B2" if status_color=="warning" else "FFCDD2" }; box-shadow: 0 4px 8px rgba(0,0,0,0.05); margin-top: 20px;">',
                unsafe_allow_html=True
            )
            st.image(overlay, caption=caption, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # --- Technical details (Collapsed) ---
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("View Technical Model Details"):
                st.json({
                    "Model Architecture": "Edge-ViT-FSL (MobileViT-XS Backbone)",
                    "Predicted Class ID": pred_idx.item(),
                    "Raw Confidence Score": f"{confidence_score:.4f}",
                    "Inference Engine": "PyTorch CPU (Optimized for Edge)",
                    "Saliency Module": "Active (Sigmoid Activation)"
                })