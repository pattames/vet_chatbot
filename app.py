import streamlit as st
import os
from bot import VeterinaryCrew
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Assistente Veterinario UNAM",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "crew" not in st.session_state:
    st.session_state.crew = None

# Initialize VeterinaryCrew
@st.cache_resource
def initialize_crew():
    """Initialize the Veterinary Crew once"""
    try:
        return VeterinaryCrew()
    except Exception as e:
        logger.error(f"Error initializing crew: {str(e)}")
        return None