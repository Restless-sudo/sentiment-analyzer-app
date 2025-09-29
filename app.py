import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# 1. पेज सेटअप
st.set_page_config(
    page_title="Sentiment & Emotion Analyzer by AMAN",
    page_icon="🎭",
    layout="wide"
)

# 2. साइडबार सेटिंग्स
st.sidebar.title("⚙️ Settings")
dark = st.sidebar.checkbox("🌙 Dark Mode", value=False)

# 3. CSS इनजेक्शन
if dark:
    st.markdown("""
    <style>
        body { background-color: #111; color: #EEE; }
        textarea, .stButton>button { background-color: #333 !important; color: #EEE !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        body { background-color: #FFF; color: #000; }
        textarea, .stButton>button { background-color: #EEE !important; color: #000 !important; }
    </style>
    """, unsafe_allow_html=True)

# 4. मॉडल लोड करें
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    try:
        return pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
    except:
        return None

emotion_model = load_emotion_model()

# 5. NLTK सेटअप
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# 6. बेसिक सेंटिमेंट फॉलबैक
positive = {"good","happy","love","great","excited","joy"}
negative = {"sad","hate","angry","terrible","depress","worst"}

def basic_sentiment(text):
    txt = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in word_tokenize(txt) if w not in stop_words]
    pos = sum(w in positive for w in tokens)
    neg = sum(w in negative for w in tokens)
    return "Positive 😊" if pos>neg else ("Negative 😞" if neg>pos else "Neutral 😐")

# 7. एनालिसिस फ़ंक्शन
def analyze(text):
    if emotion_model:
        scores = emotion_model(text)[0]
        top = max(scores, key=lambda x: x['score'])
        return f"{top['label']} ({top['score']*100:.1f}%)"
    return basic_sentiment(text)

# 8. UI लेआउट
st.markdown(f"<h1 style='text-align:center;'>🎭 Sentiment & Emotion Analyzer by AMAN</h1>", unsafe_allow_html=True)
left, right = st.columns(2, gap="large")

with left:
    st.subheader("📝 Enter your text:")
    user_text = st.text_area("", height=250, placeholder="Type your message here...")
    if st.button("🔍 Analyze Sentiment"):
        st.session_state.result = analyze(user_text)

with right:
    st.subheader("📊 Results:")
    if "result" in st.session_state:
        st.markdown(f"## {st.session_state.result}")
    else:
        st.info("Enter text and click Analyze to view results.")

# 9. Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>Made with ❤️ by AMAN</div>", unsafe_allow_html=True)
