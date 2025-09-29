import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Page config & theming
st.set_page_config(page_title="Emotion & Sentiment Analyzer", layout="wide")
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Light/dark toggle
st.sidebar.title("⚙️ Settings")
st.sidebar.checkbox("🌙 Dark Mode", key="dark_mode")
if st.session_state.dark_mode:
    st.markdown(
        """<style>
            .main { background-color:#111; color:#EEE; }
            textarea, .css-1ebnwmn, .stButton>button { background:#333; color:#EEE; }
        </style>""",
        unsafe_allow_html=True
    )

# Load emotion pipeline
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    try:
        return pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base",
                        return_all_scores=True)
    except:
        return None

emotion_model = load_emotion_model()

# Basic fallback lists
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
positive = {"good","happy","love","great","excited","joy"}
negative = {"sad","hate","angry","terrible","depress","worst"}

def basic_sentiment(text):
    text = text.lower().translate(str.maketrans("","",string.punctuation))
    words = [w for w in word_tokenize(text) if w not in stop_words]
    pos = sum(w in positive for w in words)
    neg = sum(w in negative for w in words)
    if pos>neg: return "Positive"
    if neg>pos: return "Negative"
    return "Neutral"

def analyze(text):
    # Try emotion model first
    if emotion_model:
        scores = emotion_model(text)[0]
        # Get top emotion
        top = max(scores, key=lambda x: x['score'])
        return f"{top['label']} ({top['score']*100:.1f}%)"
    # Fallback
    return basic_sentiment(text)

# UI layout
st.markdown("<div class='main'><h1>🧠 Emotion & Sentiment Analyzer</h1></div>", unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    txt = st.text_area("📝 Enter your text here", height=200)
    if st.button("Analyze"):
        result = analyze(txt)
        st.session_state.result = result

with right:
    st.markdown("### 📊 Result")
    if "result" in st.session_state:
        st.markdown(f"**{st.session_state.result}**")
    else:
        st.info("Enter text and click Analyze")

