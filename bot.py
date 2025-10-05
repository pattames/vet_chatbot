import os
from typing import List, Dict
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from vector_db import query_diseases
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# ===================================================
# AGENTS DEFINITION
# ===================================================

class VeterinaryAgents:
    """Define all agents for the veterinary chatbot system"""

    def triage_agent(self) -> Agent:
        """Agent that classifies queries and determines search necessity"""
        return Agent(
            role="Agente de Triaje Veterinario",
            goal="Clasificar consultas por tipo y urgencia, determinando si requieren búsqueda en base de conocimientos",
            backstory="""Eres un asistente veterinario experimentado en triaje de emergencias.
            Tienes la habilidad de identificar rápidamente el tipo de consulta, su urgencia médica, y determinar qué información se necesita para responder apropiadamente.""",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )


# ===================================================
# AGENTS DEFINITION
# ===================================================

class VeterinaryTasks:
    """Define all tasks for the veterinary chatbot workflow"""

    def triage_task(self, agent: Agent, user_query:str) -> Task:
        """Classify query type, urgency, and search necessity"""
        return Task(
            description=f"""Analiza esta consulta y clasificala:
            
            CONSULTA: {user_query}
            
            PASO 1 - Determina el TIPO:
            - VETERINARIA: Cualquier tema de medicina veterinaria, enfermedades, síntomas, tratamientos
            - SISTEMA: Saludos, preguntas sobre el chatbot, despedidas, agradecimientos
            - FUERA_DE_ALCANCE: Temas no veterinarios (cocina, deportes, medicina humana, etc.)

            PASO 2 - Si es de tipo VETERINARIA, determina URGENCIA:
            - EMERGENCIA: Riesgo de vida (shock, convulsiones, hemorragia severa, dificultad respiratoria)
            - URGENTE: Requiere atención pronta (vómitos persistentes, diarrea severa, dolor intenso)
            - CONSULTA: Pregunta general sobre enfermedades, tratamientos, diagnóstico
            - EDUCATIVA: Pregunta teórica o de aprendizaje

            PASO 3 - Determina si se necesita BÚSQUEDA:
            - Sí: A todas las consultas de tipo VETERINARIAS (siempre buscar información verificada)
            - No: A consultas de tipo SISTEMA Y FUERA_DE_ALCANCE

            PASO 4 - Si búsqueda = Sí, proporciona TÉRMINOS DE BÚSQUEDA en español (2 - 5 palabras clave)""",
            agent=agent,
            expected_output="""Clasificación estructurada:
            - Tipo: [VETERINARIA/SISTEMA/FUERA_DE_ALCANCE]
            - Urgencia: [EMERGENCIA/URGENTE/CONSULTA/EDUCATIVA] (solo si es de tipo VETERINARIA)
            - Búsqueda necesaria: [Sí/No]
            - Términos de búsqueda: [términos] (solo si búsqueda = Sí)"""
        )
