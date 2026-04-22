import streamlit as st
import requests
import time
from datetime import datetime
import json
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GROQ API Configuration - Read from Streamlit secrets for deployment
try:
    GROQ_API_KEY = st.secrets["Your_groq_api_key"]
except:
    # Fallback to hardcoded key for local development
    GROQ_API_KEY = "api_key"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

# Enhanced CSS with better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    
    .app-header {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .app-header h1 {
        color: #e94560;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .app-header p {
        color: #a0a0a0;
        margin-top: 0.5rem;
    }
    
    .user-msg {
        background: linear-gradient(135deg, #1e2028 0%, #252a34 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .assistant-msg {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #e94560;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .emotion-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-weight: 600;
        margin: 0.8rem 0;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    
    .joy { 
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.5);
    }
    .sadness { 
        background: linear-gradient(135deg, #2196F3 0%, #1976d2 100%);
        color: white;
        box-shadow: 0 0 20px rgba(33, 150, 243, 0.5);
    }
    .anger { 
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        box-shadow: 0 0 20px rgba(244, 67, 54, 0.5);
    }
    .fear { 
        background: linear-gradient(135deg, #9C27B0 0%, #7b1fa2 100%);
        color: white;
        box-shadow: 0 0 20px rgba(156, 39, 176, 0.5);
    }
    .love { 
        background: linear-gradient(135deg, #E91E63 0%, #c2185b 100%);
        color: white;
        box-shadow: 0 0 20px rgba(233, 30, 99, 0.5);
    }
    .surprise { 
        background: linear-gradient(135deg, #FF9800 0%, #f57c00 100%);
        color: white;
        box-shadow: 0 0 20px rgba(255, 152, 0, 0.5);
    }
    .neutral { 
        background: linear-gradient(135deg, #607D8B 0%, #455a64 100%);
        color: white;
        box-shadow: 0 0 20px rgba(96, 125, 139, 0.5);
    }
    
    .confidence {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.3rem;
        margin: 0.8rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 24px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-size: 0.9rem;
        line-height: 24px;
        font-weight: 600;
        transition: width 0.5s ease;
    }
    
    .msg-text {
        color: #e0e0e0;
        line-height: 1.7;
        white-space: pre-wrap;
        font-size: 0.95rem;
    }
    
    .timestamp {
        color: #888;
        font-size: 0.75rem;
        margin-top: 0.8rem;
        font-style: italic;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(233, 69, 96, 0.3);
        margin-bottom: 1rem;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e94560;
    }
    
    .stat-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total" not in st.session_state:
    st.session_state.total = 0
if "emotion_counts" not in st.session_state:
    st.session_state.emotion_counts = {
        "Joy": 0, "Sadness": 0, "Anger": 0, "Fear": 0,
        "Love": 0, "Surprise": 0, "Neutral": 0
    }
if "avg_confidence" not in st.session_state:
    st.session_state.avg_confidence = []

# Emotion mapping
EMOTIONS = {
    "joy": ("😊", "joy", "#4CAF50"),
    "happiness": ("😊", "joy", "#4CAF50"),
    "happy": ("😊", "joy", "#4CAF50"),
    "sadness": ("😢", "sadness", "#2196F3"),
    "sad": ("😢", "sadness", "#2196F3"),
    "anger": ("😡", "anger", "#f44336"),
    "angry": ("😡", "anger", "#f44336"),
    "fear": ("😱", "fear", "#9C27B0"),
    "scared": ("😱", "fear", "#9C27B0"),
    "anxious": ("😱", "fear", "#9C27B0"),
    "love": ("😍", "love", "#E91E63"),
    "surprise": ("😲", "surprise", "#FF9800"),
    "neutral": ("😐", "neutral", "#607D8B"),
}

def get_emotion(text):
    """Extract emotion from response"""
    text_lower = text.lower()
    for key, (emoji, css, color) in EMOTIONS.items():
        if key in text_lower:
            return emoji, css, key.capitalize(), color
    return "🎭", "neutral", "Mixed", "#607D8B"

def get_confidence(text):
    """Extract confidence percentage"""
    import re
    for line in text.split('\n'):
        if 'confidence:' in line.lower():
            nums = re.findall(r'\d+', line)
            if nums:
                return int(nums[0])
    return 75

def analyze_emotion(text):
    """Call Groq API for emotion analysis"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Analyze the emotion in this text. Be concise.

Text: "{text}"

Format (keep it SHORT):

Primary Emotion: [Joy/Sadness/Anger/Fear/Love/Surprise/Neutral]
Confidence: [0-100]%
Key Words: [2-3 emotional words from the text]
Explanation: [1-2 short sentences explaining why]

Keep it brief and clear."""

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are an emotion detection expert. Keep responses concise."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
            return f"⚠️ API Error: {error_msg}"
            
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def create_emotion_pie_chart():
    """Create pie chart of emotion distribution"""
    emotions = [k for k, v in st.session_state.emotion_counts.items() if v > 0]
    counts = [v for v in st.session_state.emotion_counts.values() if v > 0]
    
    if not emotions:
        return None
    
    colors = ['#4CAF50', '#2196F3', '#f44336', '#9C27B0', '#E91E63', '#FF9800', '#607D8B']
    
    fig = go.Figure(data=[go.Pie(
        labels=emotions,
        values=counts,
        hole=0.4,
        marker=dict(colors=colors[:len(emotions)]),
        textfont=dict(size=14, color='white')
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=13),
        height=300,
        margin=dict(t=30, b=0, l=0, r=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_confidence_timeline():
    """Create timeline of confidence scores"""
    if not st.session_state.avg_confidence:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=st.session_state.avg_confidence,
        mode='lines+markers',
        line=dict(color='#e94560', width=3),
        marker=dict(size=8, color='#e94560'),
        fill='tozeroy',
        fillcolor='rgba(233, 69, 96, 0.2)'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=250,
        margin=dict(t=30, b=30, l=40, r=20),
        xaxis=dict(title="Analysis #", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Confidence %", gridcolor='rgba(255,255,255,0.1)', range=[0, 100]),
        hovermode='x'
    )
    
    return fig

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="app-header">
        <h1>🎭 Emotion AI</h1>
        <p>NLP Extra Experiment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics
    st.markdown("### 📊 Session Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.total}</div>
            <div class="stat-label">Analyses</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_conf = int(sum(st.session_state.avg_confidence) / len(st.session_state.avg_confidence)) if st.session_state.avg_confidence else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{avg_conf}%</div>
            <div class="stat-label">Avg Conf</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Emotion distribution chart
    if st.session_state.total > 0:
        st.markdown("### 📈 Emotion Distribution")
        fig_pie = create_emotion_pie_chart()
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("### 📉 Confidence Timeline")
        fig_conf = create_confidence_timeline()
        if fig_conf:
            st.plotly_chart(fig_conf, use_container_width=True)
    
    st.markdown("---")
    
    # Emotion categories
    st.markdown("### 🎨 Emotions")
    emotions_list = [
        ("😊", "Joy", "Happy, excited"),
        ("😢", "Sadness", "Sad, grief"),
        ("😡", "Anger", "Frustrated, mad"),
        ("😱", "Fear", "Anxious, worried"),
        ("😍", "Love", "Affection, care"),
        ("😲", "Surprise", "Shocked, amazed"),
        ("😐", "Neutral", "Calm, balanced")
    ]
    
    for emoji, name, desc in emotions_list:
        st.markdown(f"**{emoji} {name}**")
        st.caption(desc)
    
    st.markdown("---")
    
    # Example prompts
    st.markdown("### 💡 Quick Examples")
    
    examples = [
        ("😊", "I got my dream job!"),
        ("😢", "I miss my grandmother."),
        ("😡", "This service is awful!"),
        ("😱", "I'm worried about the exam."),
        ("😍", "I love my family."),
        ("😲", "I can't believe I won!"),
        ("😐", "Meeting at 3 PM.")
    ]
    
    for emoji, text in examples:
        if st.button(f"{emoji} {text}", key=text, use_container_width=True):
            st.session_state.example = text
    
    st.markdown("---")
    
    # Clear and export buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total = 0
            st.session_state.emotion_counts = {k: 0 for k in st.session_state.emotion_counts}
            st.session_state.avg_confidence = []
            st.rerun()
    
    with col2:
        if st.button("💾 Export", use_container_width=True):
            chat_data = {
                "total_analyses": st.session_state.total,
                "emotion_counts": st.session_state.emotion_counts,
                "messages": st.session_state.messages
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(chat_data, indent=2),
                file_name=f"emotion_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Main content
st.markdown("""
<div class="app-header">
    <h1>💬 Emotion Detection Chat</h1>
    <p>Analyze emotions in text using Large Language Models</p>
</div>
""", unsafe_allow_html=True)

# Display chat history
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        timestamp = msg.get("time", "")
        
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-msg">
                <strong>👤 You</strong>
                <div class="msg-text">{msg["content"]}</div>
                <div class="timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            emoji, css, name, color = get_emotion(msg["content"])
            confidence = msg.get("confidence", 75)
            
            content = msg["content"].replace("\n", "<br>")
            
            st.markdown(f"""
            <div class="assistant-msg">
                <strong>🤖 Emotion AI</strong>
                <div class="emotion-badge {css}">{emoji} {name}</div>
                <div class="confidence">
                    <div class="confidence-fill" style="width: {confidence}%; background: linear-gradient(90deg, {color}, {color}dd);">
                        {confidence}% Confidence
                    </div>
                </div>
                <div class="msg-text">{content}</div>
                <div class="timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message here to analyze emotions...")

# Handle sidebar example
if "example" in st.session_state:
    user_input = st.session_state.example
    del st.session_state.example

if user_input:
    current_time = datetime.now().strftime("%I:%M %p")
    
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "time": current_time
    })
    
    with st.spinner("🔍 Analyzing emotions..."):
        response = analyze_emotion(user_input)
        confidence = get_confidence(response)
        
        st.session_state.total += 1
        st.session_state.avg_confidence.append(confidence)
        
        emoji, css, name, color = get_emotion(response)
        if name in st.session_state.emotion_counts:
            st.session_state.emotion_counts[name] += 1
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "time": current_time,
            "confidence": confidence
        })
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem;'>
    <p style='font-size: 0.9rem;'>
        🎓 <strong>NLP Continuous Assessment - Extra Experiment</strong><br>
        Text Emotion Detection using Large Language Models
    </p>
    <p style='font-size: 0.8rem; color: #666;'>
        Powered by Groq LLaMA 3.3 70B • Built with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)