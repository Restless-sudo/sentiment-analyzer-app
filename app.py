import streamlit as st
from transformers import pipeline

# 1. Page setup
st.set_page_config(
    page_title="Sentiment Analyzer by AMAN",
    page_icon="🎭",
    layout="wide"
)

# 2. Load emotion-sentiment pipeline
@st.cache_resource
def load_model():
    try:
        return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    except:
        return None

model = load_model()

# 3. Analyze function: always returns something
def analyze(text):
    if model:
        scores = model(text)[0]
        top = max(scores, key=lambda x: x["score"])
        return f"{top['label']} ({top['score']*100:.1f}%)"
    return "Neutral 😐"

# 4. UI layout
st.markdown("<h1 style='text-align:center;'>🎭 Sentiment Analyzer by AMAN</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Enter your text:")
    user_text = st.text_area("", height=200, placeholder="Type your message here…")
    if st.button("Analyze Sentiment"):
        st.session_state.result = analyze(user_text)

with col2:
    st.subheader("Results:")
    if "result" in st.session_state:
        st.markdown(f"## {st.session_state.result}")
    else:
        st.info("Enter text and click Analyze Sentiment.")

# 5. Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>Made with ❤️ by AMAN</div>", unsafe_allow_html=True)
