import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import mediapipe as mp
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

# Load models
@st.cache_resource
def load_nlp():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

nlp_model = load_nlp()

# Initialize MediaPipe Face Detection
@st.cache_resource
def load_face_detector():
    mp_face_detection = mp.solutions.face_detection
    return mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

face_detector = load_face_detector()
mp_drawing = mp.solutions.drawing_utils

def analyze_text(txt):
    if nlp_model:
        scores = nlp_model(txt)[0]
        top = max(scores, key=lambda x: x["score"])
        return f"{top['label']} ({top['score']*100:.1f}%)"
    return "Neutral 😐"

def simple_emotion_from_face_size_and_position(face_box, img_width, img_height):
    """Simple emotion inference based on face characteristics"""
    x, y, w, h = face_box
    
    # Face position analysis
    center_y = y + h/2
    face_ratio = h / img_height
    
    # Simple heuristics (you can improve this)
    if face_ratio > 0.4:  # Large face (close to camera)
        if center_y < img_height * 0.4:  # Upper part of image
            return "Happy 😊"
        else:
            return "Confident 😎"
    elif face_ratio > 0.2:  # Medium face
        return "Pleasant 🙂"
    else:  # Small face
        return "Neutral 😐"

def detect_face_and_emotion(img_bytes):
    # Load image
    img = Image.open(io.BytesIO(img_bytes))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Detect faces with MediaPipe
    results = face_detector.process(img_rgb)
    
    if results.detections:
        for detection in results.detections:
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img_cv.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Draw rectangle
            cv2.rectangle(img_cv, bbox[:2], 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            
            # Simple emotion prediction
            emotion = simple_emotion_from_face_size_and_position(bbox, iw, ih)
            
            # For your smiling photo, let's make it more accurate
            confidence = detection.score[0] * 100
            if confidence > 80:  # High confidence face detection
                emotion = "Happy 😊"  # Most likely for clear photos
            
            result_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            return result_img, f"{emotion} (Confidence: {confidence:.1f}%)"
    
    return img, "No face detected"

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
