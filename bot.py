import os
from typing import List, Dict
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from vector_db import query_diseases
import logging

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,  # Lower temperature for more consistent medical responses (adjustable)
    api_key=os.getenv("GROQ_API_KEY")
)

# ===============================================
# AGENTS DEFINITION
# ===============================================

class VeterinaryAgents:
    """Define all agents for the veterinary chatbot system"""

    def triage_agent(self) -> Agent:
        """Agent that classifies the veterinary query and determines urgency level"""
        return Agent(
           role="Agente de Triaje Veterinario",
           goal="Clasificar consultas veterinarias, identificar nivel de urgencia y determinar si se requiere información de la base de conocimientos",
           backstory="""Eres un asistente veterinario experimentado especializado en triaje.
           Tu trabajo es analizar rápidamente las consultas de los estudiantes de veterinaria y clasificarlas en:
           - EMERGENCIA: Situaciones que ponen en riesgo la vida (shock, convulciones, dificultad respiratoria severa, hemorragia grave)
           - URGENTE: Problemas que requieren atención pronta pero no inmediata (vómitos persistentes, diarrea severa, dolor intenso)
           - CONSULTA: Preguntas generales sobre enfermedades, síntomas, diagnóstico o tratamiento
           - EDUCATIVA: Preguntas teóricas o de aprendizaje
           
           También determinas si se necesita buscar información específica en la base de conocimientos.""",
           llm=llm,
           verbose=True,
           allow_delegation=False 
        )