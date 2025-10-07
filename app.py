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
    
# Header
st.title("Asistente Veterinario UNAM 🩺")
st.caption("Chatbot educativo para estudiantes de medicina veterinaria")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ Información")
    st.markdown("""
    **Sobre este asistente:**
    - Especializado en medicina veterinaria
    - Información educativa para estudiantes
    - Basado en base de conocimientos UNAM
    
    **Puede ayudarte con:**
    - Enfermedades y condiciones
    - Síntomas y diagnósticos
    - Protocolos de tratamiento
    - Emergencias veterinarias
    """)

    st.divider()

    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()

    st.markdown("""
    <div style='font-size: 0.8rem; color: #666;'>
    <strong>Nota:</strong> Este prototipo tiene límites de tokens por minuto. 
    Si ves un mensaje de error, espera unos segundos e intenta de nuevo.
    </div>
    """, unsafe_allow_html=True)

# Initialize crew if not already done
if st.session_state.crew is None:
    with st.spinner("Inicializando sistema..."):
        st.session_state.crew = initialize_crew()
        if st.session_state.crew is None:
            st.error("❌ Error al inicializar el sistema. Por favor, verifica tu configuración.")
            st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Escribe tu consulta veterinaria..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            with st.spinner("Procesando consulta..."):
                #Call the crew
                response = st.session_state.crew.run(prompt)

                # Extract the actual response text
                if hasattr(response, "raw"):
                    response_text = response.raw
                elif hasattr(response, "output"):
                    response_text = response.output
                else:
                    response_text = str(response)
                
                # Display response
                message_placeholder.markdown(response_text)

                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text
                })
        
        except Exception as e:
            error_message = str(e)

            #Check for token limit errors
            if any(keyword in error_message.lower() for keyword in ["rate limit", "token", "quota", "429"]):
                st.markdown("""
                <div class="error-box">
                    <strong>⚠️ Límite de tokens alcanzado</strong>
                    <p>El sistema ha alcanzado el límite de tokens por minuto de Groq.</p>
                    <p><strong>Por favor, espera 30-60 segundos e intenta de nuevo.</strong></p>
                    <p style='font-size: 0.85rem; margin-top: 0.5rem;'>
                    Esto es una limitación temporal debido a que el chatbot se encuentra en fase de prototipo.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                logger.warning(f"Token limit reached: {error_message}")
            else:
                # Generic error
                st.markdown(f"""
                <div class="error-box">
                    <strong>❌ Error al procesar la consulta</strong>
                    <p>{error_message}</p>
                    <p style='font-size: 0.85rem; margin-top: 0.5rem;'>
                    Por favor, intenta reformular tu pregunta o espera un momento.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                logger.error(f"Error processing query: {error_message}")

# Footer