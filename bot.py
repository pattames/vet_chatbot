import logging
import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
import vector_db
import time
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initilialize FastAPI
app = FastAPI(title="Veterinary AI Assistant", version="1.0")

# Initialize app state for sotring recent queries and clarification attempts
# Allows the chatbot to be context-aware
if not hasattr(app.state, "recent_queries"):
    app.state.recent_queries = {} # Stores last 5 queries per session (to provide context to the agents)
if not hasattr(app.state, "clarification_attempts"):
    app.state.clarification_attempts = {} # Track clarification attempts per session (helps the bot remember what question it was trying to clarify)

# Define LLM instance
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.7
)

# Define FastAPI request models
# Data model that tells FastAPI what data to expect when a request is sent to the API
class QueryRequest(BaseModel):
    query: str # Must include field called query (required)
    session_id: str = "default_session" # If user doesn't provide it, use "default_session" (to track queries from the same user)

# ===================================================
# AGENTS
# ===================================================

# Clinical Query Supervisor
clinical_supervisor = Agent(
    role="Supervisor de Consultas Clínicas Veterinarias",
    goal="Convertir consultas clínicas vagas en preguntas específicas y bien estructuradas. "
         "Identificar si la consulta describe una EMERGENCIA que requiere reconocimiento inmediato.",
    backstory="Eres un médico veterinario experimentado que ayuda a estudiantes de veterinaria "
              "a formular preguntas clínicas precisas. Reconoces signos de emergencia y priorizas "
              "la identificación de casos críticos para fines educativos.",
    llm=llm,
    verbose=True
)

supervisor_task = Task(
    description="""Analiza la consulta clínica del estudiante.
    
    **Tus tareas:**
    1. **Refinamiento de la Consulta**: Si la consulta es demasiado es vaga, refínala al:
    - Identificar síntomas e información clínica clave mencionada
    - Realizar UNA pregunta aclaratoria en caso de ser necesario (especie, edad, raza, duración de signos, signos adicionales)
    - Reestructurar la consulta para mejor búsqueda en la base de datos
    
    2. **Detección de Emergencias**: Si la consulta describe síntomas de emergencia (hemorragia severa, disnea grave, convulsiones, dilatación-vólvulo gástrico, intoxicación, colapso, anuria/disuria, etc.), marca inmediatamente el caso como EMERGENCIA.
    
   
3. **Formato de Respuesta**:
   - Para emergencias: "EMERGENCIA: [descripción breve]"
   - Para consultas refinadas: Una pregunta clínica clara y específica
   - Para aclaraciones: Una sola pregunta que termine con "?"

**Contexto reciente:** {recent_queries}
**Consulta del estudiante:** {query}

Responde SIEMPRE en español con terminología médica veterinaria apropiada.""",
    agent=clinical_supervisor,
    expected_output="Una marca de emergencia, una consulta refinada, o una pregunta de aclaración en español."
)
