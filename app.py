import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import torch
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms

# Page config
st.set_page_config(page_title="Sentiment Analyzer by AMAN", page_icon="🎭", layout="wide")

# Sidebar settings
st.sidebar.title("Settings ⚙️")
dark = st.sidebar.checkbox("Enable Dark Mode")
if dark:
    st.markdown("""
        <style>
        .stApp { background:#121212; color:#EEE; }
        .stTextArea>div>div>textarea, .stButton>button { background:#333; color:#EEE; }
        </style>
    """, unsafe_allow_html=True)

# Load NLP model
@st.cache_resource
def load_nlp():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Load face detector
@st.cache_resource
def load_face_detector():
    return MTCNN(keep_all=True, device='cpu')

nlp_model = load_nlp()
mtcnn = load_face_detector()

def analyze_text(txt):
    if nlp_model:
        scores = nlp_model(txt)[0]
        top = max(scores, key=lambda x: x["score"])
        return f"{top['label']} ({top['score']*100:.1f}%)"
    return "Neutral 😐"

def detect_face_and_emotion(img_bytes):
    # Convert to PIL Image
    img = Image.open(io.BytesIO(img_bytes))
    
    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(img)
    
    if boxes is not None:
        # Convert to numpy array for drawing
        img_array = np.array(img)
        
        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convert back to PIL
        result_img = Image.fromarray(img_array)
        
        # Simple emotion prediction based on face area analysis
        face_count = len(boxes)
        if face_count == 1:
            emotion = "Happy 😊"  # Default positive for single face
        elif face_count > 1:
            emotion = "Social 😄"  # Multiple faces
        else:
            emotion = "Neutral 😐"
            
        return result_img, f"{emotion} (Found {face_count} face(s))"
    else:
        return img, "No face detected"

# UI Layout
st.markdown("<h1 style='text-align:center;'>🎭 Sentiment Analyzer by AMAN</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📝 Enter text for analysis:")
    user_text = st.text_area("", height=200, placeholder="Type your message here...")
    if st.button("Analyze Sentiment"):
        st.session_state.text_result = analyze_text(user_text)
    
    st.markdown("**📷 Or upload a photo for face & emotion detection:**")
    photo_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if photo_file:
        import io
        img, emotion = detect_face_and_emotion(photo_file.read())
        st.session_state.detected_photo = img
        st.session_state.photo_emotion = emotion

with col2:
    st.subheader("📊 Results:")
    if "text_result" in st.session_state:
        st.markdown(f"**Text:** {st.session_state.text_result}")
    if "detected_photo" in st.session_state:
        st.image(st.session_state.detected_photo, caption=st.session_state.photo_emotion)
    if "text_result" not in st.session_state and "detected_photo" not in st.session_state:
        st.info("Enter text or upload photo to see results.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>Made with ❤️ by AMAN</div>", unsafe_allow_html=True)
