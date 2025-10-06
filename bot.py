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

    def classification_agent(self) -> Agent:
        """Agent that classifies queries and determines search necessity"""
        return Agent(
            role="Agente de Clasificación Veterinaria",
            goal="Clasificar consulta por tipo y urgencia, determinando si para responder a la consulta se requiere de una búsqueda de información",
            backstory="""Eres un asistente veterinario experimentado en la clasificación de casos.
            Tienes la habilidad de identificar rápidamente el tipo de consulta, su urgencia médica, y determinar qué información se necesita para responder apropiadamente.""",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

    def db_retrieval_agent(self) -> Agent:
        """Agent that recovers relevant information from the veterinary knowledge base"""
        return Agent(
            role="Especialista en Recuperación de Información",
            goal="Recuperar información veterinaria relevante proveniente de la base de conocimientos",
            backstory="""Eres un bibliotecario médico veterinario experto en recuperación de información.
            Sabes encontrar información precisa sobre enfermedades, tratamientos y protocolos veterinarios, proveniente de la base de conocimientos.""",
            llm=llm,
            verbose=True,
            allow_delegation=False,
            tools=[self._create_db_retrieval_tool()]
        )
    
    def _create_db_retrieval_tool(self):
        """Create tool wrapper for db information retrieval tool"""
        from crewai_tools import tool

        @tool("Recuperación de Información de Base de Conocimientos Veterinarios")
        def retrieve_db_knowledge(query: str) -> str:
            """
            Recupera información veterinaria de la base de conocimientos.

            Args:
                query: Consulta sobre enfermedades, síntomas, diagnósticos o tratamientos

            Returns:
                Información relevante de la base de conocimientos
            """
            logger.info(f"Retrieving from knowledge base: {query}")
            return query_diseases(query)
        
        return retrieve_db_knowledge

# ===================================================
# AGENTS DEFINITION
# ===================================================

class VeterinaryTasks:
    """Define all tasks for the veterinary chatbot workflow"""

    def classification_task(self, agent: Agent, user_query:str) -> Task:
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

            PASO 3 - Determina si se necesita BÚSQUEDA de información:
            - Sí: A todas las consultas de tipo VETERINARIAS
            - No: A consultas de tipo SISTEMA Y FUERA_DE_ALCANCE""",
            agent=agent,
            expected_output="""Clasificación estructurada:
            - Tipo: [VETERINARIA/SISTEMA/FUERA_DE_ALCANCE]
            - Urgencia: [EMERGENCIA/URGENTE/CONSULTA/EDUCATIVA] (solo si es de tipo VETERINARIA)
            - Búsqueda de información necesaria: [Sí/No]"""
        )
    def db_retrieval_task(self, agent: Agent, context: List[Task]) -> Task:
        """Recover knowledge base data based on triage results"""
        return Task(
            description="""Basándote en el análisis de triaje, recupera información de la base de conocimientos.

            TIENES ACCESO A LA HERRAMIENTA: "Búsqueda en Base de Conocimientos Veterinarios"

            Si el agente de clasificación indica "Búsqueda de información necesaria" = Sí:
            1. Invoca la herramienta "Búsqueda en Base de Conocimientos Veterinarios"
            2. Regresa exactamente lo que la herramienta devuelva, sin modificar

            Tu único trabajo es invocar la herramienta y pasar sus resultados al siguiente agente.""",
            agent=agent,
            expected_output="Los resultados exactos devueltos por la herramienta de búsqueda",
            context=context
        )