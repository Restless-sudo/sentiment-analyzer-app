import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import io
import time

# Page config
st.set_page_config(page_title="Advanced Sentiment Analyzer by AMAN", page_icon="🎭", layout="wide")

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

# Detailed emotion descriptions
emotion_descriptions = {
    "joy": {
        "title": "खुशी 😊",
        "description": "आप इस समय बहुत खुश और प्रसन्न हैं। यह भावना आपको ऊर्जावान महसूस कराती है और दूसरों के साथ जुड़ने की इच्छा बढ़ाती है।",
        "advice": "इस खुशी को अपने दोस्तों और परिवार के साथ शेयर करें!"
    },
    "sadness": {
        "title": "उदासी 😢",
        "description": "आप थोड़ा उदास या मन में भारीपन महसूस कर रहे हैं। यह एक प्राकृतिक भावना है जो कभी-कभी सभी को होती है।",
        "advice": "अपने किसी करीबी से बात करें या कोई अच्छी activity करें।"
    },
    "anger": {
        "title": "गुस्सा 😠", 
        "description": "आप इस समय गुस्से या चिढ़ाहट की स्थिति में हैं। यह भावना तब आती है जब कुछ हमारी अपेक्षाओं के अनुकूल नहीं होता।",
        "advice": "गहरी सांस लें और थोड़ा आराम करें। समस्या का समाधान शांति से सोचें।"
    },
    "fear": {
        "title": "डर 😨",
        "description": "आप किसी चीज़ को लेकर चिंतित या डरे हुए हैं। यह भावना हमें सुरक्षित रखने के लिए होती है।",
        "advice": "अपने डर का कारण पहचानें और किसी भरोसेमंद व्यक्ति से सलाह लें।"
    },
    "surprise": {
        "title": "आश्चर्य 😲",
        "description": "आप किसी अप्रत्याशित या नई बात से हैरान हैं। यह भावना नई चीज़ों को सीखने में मदद करती है।",
        "advice": "इस नए अनुभव का आनंद लें और जो सीखा है उसे याद रखें।"
    },
    "disgust": {
        "title": "घृणा 🤢",
        "description": "आप किसी चीज़ से नफरत या अरुचि महसूस कर रहे हैं। यह भावना हमें नुकसानदायक चीज़ों से बचाती है।",
        "advice": "उस चीज़ से दूरी बनाए रखें जो आपको परेशान कर रही है।"
    },
    "neutral": {
        "title": "सामान्य 😐",
        "description": "आप इस समय शांत और संतुलित स्थिति में हैं। कोई खास भावनात्मक उतार-चढ़ाव नहीं है।",
        "advice": "यह एक अच्छी स्थिति है। इसका फायदा उठाकर कुछ productive काम करें।"
    }
}

def analyze_text_detailed(txt):
    if nlp_model:
        scores = nlp_model(txt)[0]
        top = max(scores, key=lambda x: x["score"])
        emotion_key = top['label'].lower()
        confidence = top['score'] * 100
        
        if emotion_key in emotion_descriptions:
            emotion_info = emotion_descriptions[emotion_key]
            return {
                "emotion": emotion_info["title"],
                "confidence": confidence,
                "description": emotion_info["description"],
                "advice": emotion_info["advice"]
            }
    
    return {
        "emotion": "सामान्य 😐",
        "confidence": 50,
        "description": "आपकी भावना स्पष्ट नहीं है।",
        "advice": "अधिक विस्तार में बताएं कि आप कैसा महसूस कर रहे हैं।"
    }

# Webcam photo capture
def capture_webcam_photo():
    st.markdown("### 📷 Webcam से Photo लें:")
    
    picture = st.camera_input("अपनी तस्वीर खींचें")
    
    if picture is not None:
        # Analyze the photo
        img_analysis = analyze_photo_emotion(picture.read())
        return img_analysis
    return None

def analyze_photo_emotion(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(50, 50))
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Simple emotion detection based on facial features
            face_region = gray[y:y+h, x:x+w]
            brightness = np.mean(face_region)
            contrast = np.std(face_region)
            
            # Emotion prediction logic
            if brightness > 120 and contrast > 40:
                emotion_key = "joy"
            elif brightness < 90:
                emotion_key = "sadness"
            elif contrast > 50:
                emotion_key = "surprise"
            else:
                emotion_key = "neutral"
            
            if emotion_key in emotion_descriptions:
                emotion_info = emotion_descriptions[emotion_key]
                result_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                
                return {
                    "image": result_img,
                    "emotion": emotion_info["title"],
                    "description": emotion_info["description"],
                    "advice": emotion_info["advice"],
                    "confidence": 85
                }
    
    return {
        "image": img,
        "emotion": "No face detected",
        "description": "कोई स्पष्ट चेहरा नहीं मिला।",
        "advice": "बेहतर lighting में दोबारा कोशिश करें।",
        "confidence": 0
    }

# UI Layout
st.markdown("<h1 style='text-align:center;'>🎭 Advanced Sentiment Analyzer by AMAN</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Text और Photo दोनों से detailed emotion analysis करें</p>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "📷 Photo Upload", "🎥 Webcam Capture"])

with tab1:
    st.subheader("अपने विचार या भावनाएं लिखें:")
    user_text = st.text_area("", height=150, placeholder="जैसे: 'आज मैं बहुत खुश हूं क्योंकि मेरा interview अच्छा गया'")
    
    if st.button("🔍 Analyze करें"):
        if user_text:
            result = analyze_text_detailed(user_text)
            st.session_state.text_analysis = result

    if "text_analysis" in st.session_state:
        result = st.session_state.text_analysis
        st.markdown(f"### {result['emotion']} ({result['confidence']:.1f}%)")
        st.info(result['description'])
        st.success(f"💡 सुझाव: {result['advice']}")

with tab2:
    st.subheader("Photo Upload करें:")
    uploaded_file = st.file_uploader("अपनी तस्वीर चुनें", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        photo_result = analyze_photo_emotion(uploaded_file.read())
        st.image(photo_result["image"], caption=photo_result["emotion"])
        st.markdown(f"### {photo_result['emotion']} ({photo_result['confidence']}%)")
        st.info(photo_result['description'])
        if photo_result['advice']:
            st.success(f"💡 सुझाव: {photo_result['advice']}")

with tab3:
    st.subheader("Live Webcam से Photo:")
    webcam_result = capture_webcam_photo()
    
    if webcam_result:
        st.image(webcam_result["image"], caption=webcam_result["emotion"])
        st.markdown(f"### {webcam_result['emotion']} ({webcam_result['confidence']}%)")
        st.info(webcam_result['description'])
        if webcam_result['advice']:
            st.success(f"💡 सुझाव: {webcam_result['advice']}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>Made with ❤️ by AMAN - Advanced Emotion Intelligence</div>", unsafe_allow_html=True)
