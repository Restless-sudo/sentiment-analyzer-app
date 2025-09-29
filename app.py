import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import time

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer By AMAN",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #ffffff;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sentiment-positive {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .sentiment-negative {
        background: linear-gradient(90deg, #f44336, #da190b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .sentiment-neutral {
        background: linear-gradient(90deg, #ff9800, #f57c00);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .stTextArea > div > div > textarea {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 2px solid #4CAF50;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# NLTK data download
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))
    except:
        return set()

# Enhanced sentiment word lists
@st.cache_data
def get_sentiment_words():
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
        'love', 'like', 'happy', 'joy', 'perfect', 'best', 'brilliant', 'beautiful',
        'nice', 'pleased', 'satisfied', 'excited', 'delighted', 'thrilled', 'superb',
        'outstanding', 'marvelous', 'fabulous', 'incredible', 'impressive', 'stunning'
    }
    
    negative_words = {
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry',
        'worst', 'disgusting', 'annoying', 'frustrated', 'disappointed', 'upset',
        'depressed', 'furious', 'miserable', 'pathetic', 'useless', 'boring',
        'dreadful', 'appalling', 'atrocious', 'abysmal', 'deplorable'
    }
    
    return positive_words, negative_words

def analyze_sentiment_advanced(text, stop_words):
    positive_words, negative_words = get_sentiment_words()
    
    # Text preprocessing
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    try:
        words = word_tokenize(text)
    except:
        words = text.split()
    
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Sentiment scoring
    pos_score = sum(1 for word in words if word in positive_words)
    neg_score = sum(1 for word in words if word in negative_words)
    total_words = len(words)
    
    if total_words == 0:
        return "Neutral 😐", 0, 0, []
    
    # Calculate confidence
    confidence = ((pos_score + neg_score) / total_words) * 100
    
    # Determine sentiment
    if pos_score > neg_score:
        sentiment = "Positive 😊"
        sentiment_class = "positive"
    elif neg_score > pos_score:
        sentiment = "Negative 😞"
        sentiment_class = "negative"
    else:
        sentiment = "Neutral 😐"
        sentiment_class = "neutral"
    
    return sentiment, pos_score, neg_score, confidence, sentiment_class

def main():
    # Load NLTK data
    stop_words = download_nltk_data()
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Sentiment Analyzer By AMAN</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Create perfect 50-50 columns
    left_col, right_col = st.columns([1, 1], gap="large")
    
    # Left Column - Input Section
    with left_col:
        st.markdown("### 📝 Enter your text:")
        
        # Text input with placeholder
        user_input = st.text_area(
            "",
            height=300,
            placeholder="Type your message here... For example:\n• I am feeling great today!\n• This movie was terrible\n• The weather is okay",
            help="Enter any text to analyze its sentiment"
        )
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button(
                "🔍 Analyze Sentiment", 
                type="primary", 
                use_container_width=True
            )
        
        # Sample text buttons
        st.markdown("**Quick samples:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("😊 Positive Sample"):
                st.session_state.sample_text = "I love this new phone! The camera quality is amazing and the battery life is fantastic. Highly recommend!"
        with col2:
            if st.button("😞 Negative Sample"):
                st.session_state.sample_text = "This service is terrible. Very disappointed and frustrated with the poor quality."
    
    # Right Column - Results Section
    with right_col:
        st.markdown("### 📊 Results:")
        
        # Handle sample text
        if hasattr(st.session_state, 'sample_text'):
            user_input = st.session_state.sample_text
            analyze_btn = True  # Auto-analyze sample
            del st.session_state.sample_text
        
        if analyze_btn and user_input.strip():
            # Show loading animation
            with st.spinner('Analyzing sentiment...'):
                time.sleep(0.5)  # Small delay for effect
                
            # Perform analysis
            sentiment, pos_score, neg_score, confidence, sentiment_class = analyze_sentiment_advanced(user_input, stop_words)
            
            # Display sentiment result with custom styling
            if sentiment_class == "positive":
                st.markdown(f'<div class="sentiment-positive">{sentiment}</div>', unsafe_allow_html=True)
            elif sentiment_class == "negative":
                st.markdown(f'<div class="sentiment-negative">{sentiment}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="sentiment-neutral">{sentiment}</div>', unsafe_allow_html=True)
            
            # Metrics in columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="😊 Positive",
                    value=pos_score,
                    delta=f"{(pos_score/len(user_input.split()))*100:.1f}%"
                )
            
            with metric_col2:
                st.metric(
                    label="😞 Negative", 
                    value=neg_score,
                    delta=f"{(neg_score/len(user_input.split()))*100:.1f}%"
                )
            
            with metric_col3:
                st.metric(
                    label="🎯 Confidence",
                    value=f"{confidence:.1f}%"
                )
            
            # Text analysis details
            st.markdown("**📋 Analysis Details:**")
            st.info(f"**Text Length:** {len(user_input)} characters, {len(user_input.split())} words")
            
            # Your analyzed text
            with st.expander("📄 Your Text", expanded=False):
                st.write(user_input)
            
        else:
            # Placeholder when no analysis
            st.info("👆 Enter some text and click 'Analyze Sentiment' to see results!")
            
            # Instructions
            st.markdown("""
            **How it works:**
            1. Type or paste your text in the left box
            2. Click the 'Analyze Sentiment' button
            3. See instant results here!
            
            **Features:**
            - ✅ Real-time sentiment analysis
            - ✅ Confidence scoring
            - ✅ Word-level breakdown
            - ✅ Sample text options
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>Made with ❤️ by AMAN | Powered by Streamlit & NLTK</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
