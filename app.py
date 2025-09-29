import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from fer import FER
import cv2
import numpy as np

# 1. पेज सेटअप
st.set_page_config(
    page_title="Sentiment Analyzer by AMAN",
    page_icon="🎭",
    layout="wide"
)

# 2. सेटिंग्स साइडबार
st.sidebar.title("Settings ⚙️")
dark_mode = st.sidebar.checkbox("Enable Dark Mode")

# 3. CSS theming
if dark_mode:
    st.markdown("""
        <style>
        .stApp { background:#121212; color:#EEE; }
        .stTextArea>div>div>textarea, .stButton>button { background:#333; color:#EEE; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background:#FAFAFA; color:#111; }
        .stTextArea>div>div>textarea, .stButton>button { background:#FFF; color:#111; }
        </style>
    """, unsafe_allow_html=True)

# 4. NLP मॉडल लोड करें
@st.cache_resource
def load_nlp_model():
    try:
        return pipeline("text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        return_all_scores=True)
    except:
        return None
nlp_model = load_nlp_model()

# 5. NLTK सेटअप
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

positive_words = {"good","happy","love","great","joy"}
negative_words = {"sad","hate","angry","terrible","worst"}

def basic_sentiment(text):
    text = text.lower().translate(str.maketrans("","",string.punctuation))
    tokens = [w for w in word_tokenize(text) if w not in stop_words]
    pos = sum(w in positive_words for w in tokens)
    neg = sum(w in negative_words for w in tokens)
    if pos > neg:
        return "Positive 😊"
    elif neg > pos:
        return "Negative 😞"
    else:
        return "Neutral 😐"

def analyze_text(text):
    if nlp_model:
        scores = nlp_model(text)[0]
        top = max(scores, key=lambda x: x["score"])
        return f"{top['label']} ({top['score']*100:.1f}%)"
    else:
        return basic_sentiment(text)

# 6. FER emotion detector
detector = FER(mtcnn=True)
def analyze_photo(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    faces = detector.detect_emotions(img)
    if not faces:
        return "No face detected"
    top_emotion, top_score = max(faces[0]["emotions"].items(), key=lambda x: x[1])
    return f"{top_emotion.title()} ({top_score*100:.1f}%)"

# 7. UI लेआउट
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
        st.session_state.photo_result = analyze_photo(photo_file.read())

with col2:
    st.subheader("📊 Results:")
    if "text_result" in st.session_state:
        st.markdown(f"**Text:** {st.session_state.text_result}")
    if "photo_result" in st.session_state:
        st.markdown(f"**Photo:** {st.session_state.photo_result}")
    if "text_result" not in st.session_state and "photo_result" not in st.session_state:
        st.info("Enter text or upload photo to see results.")

# 8. Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>Made with ❤️ by AMAN</div>", unsafe_allow_html=True)
