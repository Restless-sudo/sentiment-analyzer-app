import streamlit as st
from streamlit_elements import elements, mui, html
from transformers import pipeline
import cv2, numpy as np
from PIL import Image

st.set_page_config(layout="wide", page_title="Pro Sentiment Analyzer")
st.markdown("""
<style>
body { font-family:'Segoe UI', sans-serif; }
.card { padding:16px; border-radius:8px; background:#ffffffaa; box-shadow:0 2px 8px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_nlp(): return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
nlp = load_nlp()

# Layout with streamlit-elements
with elements("demo"):
    mui.Box(style={"margin":"20px"})[
        mui.Typography("🎭 Pro Sentiment Analyzer by AMAN", variant="h3", gutterBottom=True)
    ]
    mui.Grid(container=True, spacing=4)[
        mui.Grid(item=True, xs=6)[
            mui.Card(className="card")[
                mui.Typography("📝 Text Analysis", variant="h5"),
                mui.TextField(id="input-text", label="Type mood...", fullWidth=True),
                mui.Button("Analyze", variant="contained", color="primary", id="btn-text"),
                html.DIV(id="output-text")
            ]
        ],
        mui.Grid(item=True, xs=6)[
            mui.Card(className="card")[
                mui.Typography("📷 Photo Emotion", variant="h5"),
                html.INPUT(type="file", accept="image/*", id="file-photo"),
                html.DIV(id="output-photo")
            ]
        ]
    ]

# Callbacks
def analyze_text(text):
    scores = nlp(text)[0]
    top = max(scores, key=lambda x: x["score"])
    return top["label"], top["score"]

def render_output_text(label, score):
    bar = st.progress(score)
    st.markdown(f"**Emotion:** {label}  \n**Confidence:** {score*100:.1f}%")

def render_output_photo(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    st.image(img, use_column_width=True)

# Wire up events
if st.session_state.get("btn-text_clicked"):
    lbl, sc = analyze_text(st.session_state["input-text"])
    render_output_text(lbl, sc)

uploaded = st.file_uploader("", type=["jpg","png"])
if uploaded:
    render_output_photo(uploaded.read())
