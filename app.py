import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import io

# 1. पेज कॉन्फ़िग
st.set_page_config(
    page_title="Pro Sentiment Analyzer by AMAN",
    page_icon="🎭",
    layout="wide"
)

# 2. कस्टम थीमिंग (CSS)
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
section.main .block-container { padding: 2rem; }
.card { background: #fff; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }
.card h3 { margin-top: 0; }
.progress-bar { width: 100%; background: #eee; border-radius: 4px; overflow: hidden; height: 12px; margin: 0.5rem 0; }
.progress { height: 100%; background: #4caf50; }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar सेटिंग
st.sidebar.title("Settings ⚙️")
dark = st.sidebar.checkbox("Enable Dark Mode")
if dark:
    st.markdown("""
    <style>
      body, .block-container { background: #121212; color: #EEE; }
    </style>
    """, unsafe_allow_html=True)

# 4. NLP मॉडल लोड
@st.cache_resource
def load_nlp():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
nlp = load_nlp()

def analyze_text(text):
    scores = nlp(text)[0]
    top = max(scores, key=lambda x: x["score"])
    return top["label"].capitalize(), top["score"]

# 5. फोटो एनालिसिस (साफ़-सुथरा face detection + simple emotion)
def detect_face_and_emotion(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    emotion = "No face"
    confidence = 0.0
    if len(faces) > 0:
        x,y,w,h = faces[0]
        cv2.rectangle(arr, (x,y), (x+w,y+h), (0,255,0), 2)
        face = gray[y:y+h, x:x+w]
        bright = np.mean(face)
        if bright > 120: emotion, confidence = "Happy", 0.9
        else: emotion, confidence = "Neutral", 0.7
    return Image.fromarray(arr), emotion, confidence

# 6. UI लेआउट
st.markdown("<h1 style='text-align:center;'>🎭 Pro Sentiment Analyzer by AMAN</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📝 Text Analysis", "📷 Photo Analysis"])

with tab1:
    st.markdown("<div class='card'><h3>Text Sentiment</h3></div>", unsafe_allow_html=True)
    txt = st.text_area("Enter your message...", height=150)
    if st.button("Analyze Text"):
        if txt:
            label, score = analyze_text(txt)
            st.markdown(f"<div class='card'><h3>{label}</h3></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='progress-bar'><div class='progress' style='width:{score*100:.1f}%;'></div></div>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {score*100:.1f}%")

with tab2:
    st.markdown("<div class='card'><h3>Photo Emotion</h3></div>", unsafe_allow_html=True)
    file = st.file_uploader("Upload a photo", type=["jpg","jpeg","png"])
    if file:
        img, emotion, conf = detect_face_and_emotion(file.read())
        st.image(img, use_column_width=True)
        st.markdown(f"<div class='card'><h3>{emotion}</h3><p>Confidence: {conf*100:.1f}%</p></div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:gray;'>Made with ❤️ by AMAN</div>", unsafe_allow_html=True)
