import streamlit as st
from transformers import pipeline
from deepface import DeepFace
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import cv2
import numpy as np

# 1. Page config
st.set_page_config(
    page_title="Sentiment Analyzer by AMAN",
    page_icon="🎭",
    layout="wide"
)

# 2. Sidebar: Dark mode
st.sidebar.title("Settings ⚙️")
dark = st.sidebar.checkbox("Enable Dark Mode")
if dark:
    st.markdown("""
    <style>
      .stApp { background:#121212; color:#EEE; }
      textarea, .stButton>button { background:#333; color:#EEE; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
      .stApp { background:#FAFAFA; color:#111; }
      textarea, .stButton>button { background:#FFF; color:#111; }
    </style>""", unsafe_allow_html=True)

# 3. Load NLP model
@st.cache_resource
def load_nlp():
    try:
        return pipeline("text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        return_all_scores=True)
    except:
        return None

nlp_model = load_nlp()

# 4. NLTK setup
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# 5. Basic sentiment fallback
pos_words = {"good","happy","love","great","joy"}
neg_words = {"sad","hate","angry","terrible","worst"}
def basic_sentiment(txt):
    t = txt.lower().translate(str.maketrans("","",string.punctuation))
    toks = [w for w in word_tokenize(t) if w not in stop_words]
    p = sum(w in pos_words for w in toks)
    n = sum(w in neg_words for w in toks)
    return "Positive 😊" if p>n else ("Negative 😞" if n>p else "Neutral 😐")

# 6. Text analysis
def analyze_text(txt):
    if nlp_model:
        scores = nlp_model(txt)[0]
        top = max(scores, key=lambda x: x["score"])
        return f"{top['label']} ({top['score']*100:.1f}%)"
    return basic_sentiment(txt)

# 7. Photo analysis
def analyze_photo(img_bytes):
    # Read into OpenCV
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # DeepFace emotion detection
    try:
        result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
        return result["dominant_emotion"].title()
    except:
        return "No face detected"

# 8. UI Layout
st.markdown("<h1 style='text-align:center;'>🎭 Sentiment Analyzer by AMAN</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📝 Enter your text:")
    txt = st.text_area("", height=200, placeholder="Type your message here…")
    if st.button("Analyze Sentiment"):
        st.session_state.text_result = analyze_text(txt)

    st.markdown("**📷 Or upload a photo:**")
    photo = st.file_uploader("", type=["jpg","png"], accept_multiple_files=False)
    if photo:
        emotion = analyze_photo(photo.read())
        st.session_state.photo_result = emotion

with col2:
    st.subheader("📊 Results:")
    if "text_result" in st.session_state:
        st.markdown(f"**Text:** {st.session_state.text_result}")
    if "photo_result" in st.session_state:
        st.markdown(f"**Photo:** {st.session_state.photo_result}")

# 9. Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>Made with ❤️ by AMAN</div>", unsafe_allow_html=True)
