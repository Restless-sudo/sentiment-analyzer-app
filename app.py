import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer by AMAN",
    page_icon="🎭",
    layout="wide"
)

st.sidebar.title("Settings ⚙️")
dark = st.sidebar.checkbox("Enable Dark Mode")
if dark:
    st.markdown("""
        <style>
        .stApp { background:#121212; color:#EEE; }
        .stTextArea>div>div>textarea, .stButton>button { background:#333; color:#EEE; }
        </style>
    """, unsafe_allow_html=True)

# Text Emotion Setup
@st.cache_resource
def load_nlp():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
nlp_model = load_nlp()

def analyze_text(txt):
    if nlp_model:
        scores = nlp_model(txt)[0]
        top = max(scores, key=lambda x: x["score"])
        return f"{top['label']} ({top['score']*100:.1f}%)"
    return "Neutral 😐"

# Face Detection Demo (No deep emotion)
def detect_face(img_bytes):
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return im_pil, len(faces)

# Layout
st.markdown("<h1 style='text-align:center;'>🎭 Sentiment Analyzer by AMAN</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("📝 Enter text for analysis:")
    user_text = st.text_area("", height=200, placeholder="Type your message here...")
    if st.button("Analyze Sentiment"):
        st.session_state.text_result = analyze_text(user_text)
    st.markdown("**📷 Or upload a photo for face detection:**")
    photo_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if photo_file:
        img, fc = detect_face(photo_file.read())
        st.session_state.detected_photo = img
        st.session_state.face_count = fc

with col2:
    st.subheader("📊 Results:")
    if "text_result" in st.session_state:
        st.markdown(f"**Text:** {st.session_state.text_result}")
    if "detected_photo" in st.session_state:
        st.image(st.session_state.detected_photo, caption=f"Faces Detected: {st.session_state.face_count}")
    if "text_result" not in st.session_state and "detected_photo" not in st.session_state:
        st.info("Enter text or upload photo to see results.")
st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>Made with ❤️ by AMAN</div>", unsafe_allow_html=True)
