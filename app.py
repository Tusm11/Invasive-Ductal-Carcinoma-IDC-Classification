import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
import pickle

# Set Page Config
st.set_page_config(
    page_title="Breast Cancer AI Diagnostic Tool",
    page_icon="🩺",
    layout="wide"
)

# --- CSS for Professional Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    
    /* Customizing the File Uploader to look like a premium Drop Zone */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #1E3A8A !important;
        border-radius: 15px !important;
        padding: 40px !important;
        background-color: #ffffff !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3b82f6 !important;
        background-color: #f0f7ff !important;
    }

    .stAlert {
        border-radius: 10px;
    }
    .predict-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #1E3A8A;
        font-weight: 800;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_resource
def load_model_from_file(model_type):
    """Load model from .pkl or .h5 based on availability"""
    model_path_pkl = f"models/{model_type}_model.pkl"
    model_path_h5 = f"models/{model_type}_best.h5"
    
    try:
        if os.path.exists(model_path_pkl):
            with open(model_path_pkl, 'rb') as f:
                model = pickle.load(f)
            return model
        elif os.path.exists(model_path_h5):
            model = tf.keras.models.load_model(model_path_h5)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_image(uploaded_file):
    """Convert streamlit uploader file to numpy array and preprocess"""
    image = Image.open(uploaded_file)
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize to 50x50 as expected by the model
    image_resized = image.resize((50, 50))
    img_array = keras.utils.img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0, image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Generate Grad-CAM heatmap"""
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0] # Binary classification assumes IDC Positive is probability

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    img_array = keras.utils.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img_array
    return keras.utils.array_to_img(superimposed_img)

# --- UI Layout ---

st.title("🩺 Breast Cancer Classification using Deep Learning")
st.markdown("""
This application uses **Histopathology Deep Learning** to detect Invasive Ductal Carcinoma (IDC). 
The models are trained on microscopic breast tissue patches to distinguish between benign and malignant cells.
""")

st.warning("⚠️ **Disclaimer:** This tool is for research and educational purposes only. It should not be used for actual clinical diagnosis.")

# Sidebar
st.sidebar.header("🔧 Settings & Metrics")
model_choice = st.sidebar.selectbox("Select Model Architecture", ["vgg16", "resnet50"])

if st.sidebar.button("Load Selected Model"):
    with st.spinner("Loading AI parameters..."):
        st.session_state.model = load_model_from_file(model_choice)
        st.session_state.current_model_type = model_choice
        if st.session_state.model:
            st.sidebar.success(f"{model_choice.upper()} Loaded Successfully!")
        else:
            st.sidebar.error("Model files not found. Please train the models first.")

# Metrics Display (Fixed metrics from evaluation)
if model_choice == "vgg16":
    st.sidebar.markdown("### 📊 Model Performance (VGG16)")
    st.sidebar.metric("Accuracy", "76.0%")
    st.sidebar.metric("Sensitivity (Recall)", "78.0%")
    st.sidebar.metric("AUC Score", "0.840")
else:
    st.sidebar.markdown("### 📊 Model Performance (ResNet50)")
    st.sidebar.metric("Accuracy", "74.0%")
    st.sidebar.metric("Sensitivity (Recall)", "65.0%")
    st.sidebar.metric("AUC Score", "0.761")

# Main Section
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Step 1: Upload Image")
    st.info("💡 **Tip:** You can drag any `.png` file from your `data/` folder and drop it into the box below.")
    
    uploaded_file = st.file_uploader(
        "Drag & Drop Histopathology Patch Here", 
        type=["jpg", "png", "jpeg"],
        help="Target size is 50x50 pixels. Individual patches only."
    )
    
    # --- PROACTIVE ADDITION: Load Sample Button ---
    st.write("--- or ---")
    if st.button("📂 Load Random Sample from Dataset"):
        import glob
        import random
        # Search for any PNG in the histopathology data
        sample_paths = glob.glob("data/breast_histopathology/**/*.png", recursive=True)
        if sample_paths:
            random_sample = random.choice(sample_paths)
            # Simulate an uploaded file object
            with open(random_sample, "rb") as f:
                st.session_state.sample_img_bytes = f.read()
                st.session_state.sample_img_name = os.path.basename(random_sample)
            st.success(f"Loaded: {st.session_state.sample_img_name}")
        else:
            st.error("No samples found in data folder. Is the dataset downloaded?")

    # Handle either uploaded or sample image
    display_img = None
    if uploaded_file:
        input_tensor, display_img = process_image(uploaded_file)
    elif "sample_img_bytes" in st.session_state:
        from io import BytesIO
        sample_io = BytesIO(st.session_state.sample_img_bytes)
        input_tensor, display_img = process_image(sample_io)

    if display_img:
        st.image(display_img, caption="Analyze this Patch", use_container_width=True)

with col2:
    st.header("🧪 Step 2: Prediction")
    # Check if we have an image to predict (either uploaded or sample)
    has_image = uploaded_file or "sample_img_bytes" in st.session_state
    
    if has_image and "model" in st.session_state and st.session_state.model:
        if st.button("🚀 Predict Diagnosis"):
            with st.spinner("Analyzing tissue patterns..."):
                # Prediction
                prediction = st.session_state.model.predict(input_tensor)[0][0]
                label = "Malignant (IDC Positive)" if prediction > 0.5 else "Benign (IDC Negative)"
                color = "red" if prediction > 0.5 else "green"
                
                st.markdown(f"### Result: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                
                # Probability
                prob = prediction if prediction > 0.5 else (1 - prediction)
                st.write(f"Confidence Level: {prob*100:.2f}%")
                st.progress(float(prob))
                
                # Explanation
                if prediction > 0.5:
                    st.info("The model detected high probability of cellular patterns associated with Invasive Ductal Carcinoma.")
                else:
                    st.success("The model detected patterns consistent with benign tissue.")

                # Grad-CAM Section
                st.divider()
                st.header("🔍 Explainability (Grad-CAM)")
                
                # Determine layer name
                layer_name = "block5_conv3" if st.session_state.current_model_type == "vgg16" else "conv5_block3_out"
                
                try:
                    heatmap = make_gradcam_heatmap(input_tensor, st.session_state.model, layer_name)
                    super_img = superimpose_heatmap(display_img, heatmap)
                    
                    gc_col1, gc_col2 = st.columns(2)
                    with gc_col1:
                        st.image(display_img, caption="Original", use_container_width=True)
                    with gc_col2:
                        st.image(super_img, caption="Activation Focus", use_container_width=True)
                    
                    st.caption("The heatmap highlights the regions in the image that most heavily influenced the AI's classification decision.")
                except Exception as e:
                    st.warning(f"Grad-CAM could not be generated for this request: {e}")

    elif not uploaded_file:
        st.write("Please upload an image to begin.")
    elif "model" not in st.session_state:
        st.error("Please load the model from the sidebar first.")

st.sidebar.markdown("---")
st.sidebar.info("Designed by M V Abhiram | AI/ML Research")
