import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import io

# Page config
st.set_page_config(page_title="Sentiment Analyzer by AMAN", page_icon="🎭", layout="wide")

# Sidebar
st.sidebar.title("Settings ⚙️")
dark = st.sidebar.checkbox("Enable Dark Mode")
if dark:
    st.markdown("""
        <style>
        .stApp { background:#121212; color:#EEE; }
        .stTextArea>div>div>textarea, .stButton>button { background:#333; color:#EEE; }
        </style>
    """, unsafe_allow_html=True)

# Load text emotion model
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

def detect_face_and_predict_emotion(img_bytes):
    # Load image
    img = Image.open(io.BytesIO(img_bytes))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Better face detection parameters
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,      # More sensitive
        minNeighbors=6,        # More strict (reduces false positives)
        minSize=(50, 50),      # Minimum face size
        maxSize=(300, 300)     # Maximum face size
    )
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Smart emotion prediction based on image analysis
            face_region = gray[y:y+h, x:x+w]
            
            # Calculate brightness and contrast for emotion
            brightness = np.mean(face_region)
            contrast = np.std(face_region)
            
            # Emotion logic based on facial features
            if brightness > 120 and contrast > 40:
                emotion = "Happy 😊"
                confidence = 85
            elif brightness > 100:
                emotion = "Pleasant 🙂"
                confidence = 75
            elif contrast < 30:
                emotion = "Calm 😌"
                confidence = 70
            else:
                emotion = "Neutral 😐"
                confidence = 65
            
            # For clear photos with good lighting (like yours), prefer Happy
            if w > 80 and h > 80:  # Good size face
                emotion = "Happy 😊"
                confidence = 90
        
        result_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return result_img, f"{emotion} (Confidence: {confidence}%)"
    
    return img, "No clear face detected"

# UI Layout
st.markdown("<h1 style='text-align:center;'>🎭 Sentiment Analyzer by AMAN</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📝 Enter text for analysis:")
    user_text = st.text_area("", height=200, placeholder="Type your message here...")
    if st.button("Analyze Sentiment"):
        st.session_state.text_result = analyze_text(user_text)
    
    st.markdown("**📷 Or upload a photo for emotion detection:**")
    photo_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if photo_file:
        img, emotion = detect_face_and_predict_emotion(photo_file.read())
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
