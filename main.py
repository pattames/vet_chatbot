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
    model="groq/llama-3.3-70b-versatile",
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
    
    def veterinary_specialist_agent(self) -> Agent:
        """Agent that formulates veterinary responses"""
        return Agent(
            role="Veterinario Clínico Educador",
            goal="Proporcionar respuestas veterinarias educativas, precisas y apropiadas para estudiantes",
            backstory="""Eres un veterinario clínico senior con más de 15 años de experiencia y pasión por la enseñanza.
            Te especializas en medicina de pequeños animales y eres excelente explicando conceptos complejos de manera clara. Siempre priorizas la seguridad del paciente y la precisión médica.""",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
    
    def quality_control_agent(self) -> Agent:
        """Agent that verifies response safety and quality"""
        return Agent(
            role="Supervisor de Calidad y Seguridad",
            goal="Verificar que las respuestas sean seguras, precisas y apropiadas a nivel educativo",
            backstory="""Eres un supervisor de educación veterinaria enfocado en seguridad del paciente.
            Revisas meticulosamente la información médica para asegurar que sea precisa, segura y apropiada para estudiantes de veterinaria.""",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
    
    def _create_db_retrieval_tool(self):
        """Create tool wrapper for db information retrieval tool"""
        from crewai.tools import BaseTool
        from typing import Type
        from pydantic import BaseModel, Field

        class SearchInput(BaseModel):
            """Input schema for knowledge base search"""
            query: str = Field(..., description="Consulta refinada sobre enfermedades, síntomas, diagnósticos o tratamientos veterinarios")

        class DbRetrievalTool(BaseTool):
            name: str = "Recuperación de Información de Base de Conocimientos Veterinarios"
            description: str = "Recuperación de información veterinaria relevante proveniente de la base de conocimientos"
            args_schema: Type[BaseModel] = SearchInput

            def _run(self, query: str) -> str:
                return query_diseases(query)
            
        return DbRetrievalTool()
            
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
            - EMERGENCIA: Riesgo de vida inmediato (shock, convulsiones, hemorragia severa, dificultad respiratoria severa, intoxicaciones, etc.)
            - NO_EMERGENCIA: Todas las demás consultas veterinarias

            PASO 3 - Determina si se necesita BÚSQUEDA de información:
            - Sí: A todas las consultas de tipo VETERINARIAS
            - No: A consultas de tipo SISTEMA Y FUERA_DE_ALCANCE
            
            PASO 4 - Si búsqueda = Sí, crea CONSULTA REFINADA:
            - Reformula la consulta del usuario para búsqueda semántica
            - Usa frases completas con contexto médico
            - Ejemplos:
                • "Mi perro comió chocolate" → "intoxicación por chocolate en perros"
                • "Perro con vómitos y diarrea con sangre" → "síntomas de vómitos y diarrea hemorrágica en perros"
                • "¿Qué es parvovirus?" → "Información sobre el parvovirus canino"
            - Mantén contexto que sea importante (síntomas, especie, urgencia)
            - NO uses solo palabras clave sueltas""",
            agent=agent,
            expected_output="""Clasificación estructurada:
            - Tipo: [VETERINARIA/SISTEMA/FUERA_DE_ALCANCE]
            - Urgencia: [EMERGENCIA/NO_EMERGENCIA] (solo si es de tipo VETERINARIA)
            - Búsqueda de información necesaria: [Sí/No]
            - Consulta refinada: [frase completa con contexto] (solo si búsqueda = Sí)"""
        )

    def db_retrieval_task(self, agent: Agent, context: List[Task]) -> Task:
        """Recover knowledge base data based on triage results"""
        return Task(
            description="""Basándote en el análisis del agente de clasificación, recupera información de la base de conocimientos.

            TIENES ACCESO A LA HERRAMIENTA: "Recuperación de Información de Base de Conocimientos Veterinarios"

            Si el agente de clasificación indica "Búsqueda de información necesaria" = No:
            - Regresa exactamente: "BÚSQUEDA NO REQUERIDA"

            Si el agente de clasificación indica "Búsqueda de información necesaria" = Sí:
            1. Recupera la "Consulta refinada" del agente de clasificación
            2. Invoca la herramienta "Recuperación de Información de Base de Conocimientos Veterinarios" utilizando la consulta refinada como argumento. Es importante que utilices la consulta refinada completa, no palabras clave
            3. Regresa exactamente lo que la herramienta devuelva, sin modificar

            Tu único trabajo es invocar la herramienta y pasar sus resultados al siguiente agente.""",
            agent=agent,
            expected_output="""Uno de los siguientes:
            - "BÚSQUEDA NO REQUERIDA" (si el agente de clasificación indicó que no se necesita buscar)
            - Los resultados exactos devueltos por la herramienta de búsqueda""",
            context=context
        )
    
    def specialist_response_task(self, agent: Agent, user_query: str, context: List[Task]) -> Task:
        """Formulate appropriate response based on query type"""
        return Task(
            description=f"""Basándote en la clasificación de la consulta y la información recuperada (en caso de que hubiera), formula una respuesta apropiada.
            
            CONSULTA ORIGINAL: {user_query}

            TIPO 1: CONSULTAS VETERINARIAS
            A) Si hay información proveniente de la base de conocimientos:
                - Úsala como fuente principal
                - Incluye detalles específicos (dosis, protocolos, valores diagnósticos)
                - Si es EMERGENCIA, comienza con: ⚠️ EMERGENCIA VETERINARIA
            B) Si NO hay información proveniente de la base de conocimientos:
                - Comienza con: "⚠️ Información basada en conocimiento general (no verificado en base de conocimientos de la UNAM):"
                - Evitar dosis específicas a menos de que vengan de fuentes verificadas
                - Sugiere consultar literatura veterinaria adicional
            Estructura: [Alerta de emergencia si aplica] + Respuesta principal + Detalles

            TIPO 2: CONSULTAS DE SISTEMA
            A) Saludos/¿Qué puedes hacer?:
                "¡Hola! Soy tu asistente de aprendizaje en medicina veterinaria 🩺.

                Puedo ayudarte con:
                • Enfermedades y condiciones veterinarias
                • Síntomas y diagnósticos
                • Protocolos de tratamiento
                • Emergencias veterinarias
                • Procedimientos y anestesia

                ¿En qué tema veterinario te gustaría que te ayude?"
            B) Despedidas:
                "¡Hasta pronto! Estoy aquí cuando necesites ayuda con temas veterinarios 🐕🐈"
            C) Agradecimientos: "¡Con gusto! Si tienes más consultas veterinarias, estaré encantado de ayudarte 😊."

            TIPO 3: CONSULTAS FUERA DE ALCANCE
            "Soy un asistente especializado en medicina veterinaria.
            
            Puedo ayudarte con preguntas sobre enfermedades, síntomas, diagnósticos y tratamientos veterinarios, pero no puedo asistir con [mención breve del tema].

            Tienes alguna consulta veterinaria en la que pueda ayudarte?"

            Para todos los tipos de respuesta mantén un tono profesional pero accesible.""",
            agent=agent,
            expected_output="Respuesta completa y apropiada para el tipo de consulta (veterinaria, no veterinaria o de sistema)",
            context=context
        )
    
    def quality_check_task(self, agent: Agent, context: List[Task]) -> Task:
        """Review response for safety, accuracy, and quality"""
        return Task(
            description="""Revisa la respuesta del Veterinario Clínico Educador y asegura su calidad.
            
            IMPORTANTE: Tu respuesta final debe contener ÚNICAMENTE la respuesta del Veterinario Clínico Educador (con modificaciones solo si es necesario). NO incluyas información de clasificación ni análisis de otros agentes.
            
            Únicamente para respuestas a consultas de tipo VETERINARIAS verifica los siguientes puntos:
            - SEGURIDAD:
                - Emergencias claramente marcadas con ⚠️ EMERGENCIA VETERINARIA
                - Dosis/protocolos correctos
                - NO hay dosis específicas sin fuente verificada
                - Advertencias apropiadas sobre riesgos
            - TRANSPARENCIA DE FUENTE:
                - Información proveniente de la base de conocimientos se usa sin modificar
                - Información proveniente de conocimiento general marcada con "⚠️ Información basada en conocimiento general (no verificado en base de conocimientos de la UNAM)"
            - CALIDAD EDUCATIVA:
                - Terminología médica correcta en español
                - Explicaciones claras para estudiantes
            - DISCLAIMER OBLIGATORIO (agregar al final):
                "📚 Nota Educativa: Esta información es para fines educativos. En la práctica clínica, cada caso debe evaluarse individualmente considerando el historial completo, examen físico y resultados diagnósticos."
            
            Para respuestas a consultas de tipo SISTEMA o tipo FUERA_DE_ALCANCE:
            - NO hagas cambios
            - NO agregues el disclaimer
            - Regresa ÚNICAMENTE la respuesta del Veterinario Clínico Educador, sin agregar información de clasificación ni de otros agentes.""",
            agent=agent,
            expected_output="""ÚNICAMENTE la respuesta del Veterinario Clínico Educador (revisada y corregida si es tipo VETERINARIA, o sin cambios si es tipo SISTEMA o FUERA_DE_ALCANCE). NO incluyas información de clasificación.""",
            context=context
    )

# ===================================================
# CREW ORCHESTRATION
# ===================================================
class VeterinaryCrew:
    """Orchestrate the multi-agent veterinary chatbot workflow"""

    def __init__(self):
        self.agent_manager = VeterinaryAgents()
        self.task_manager = VeterinaryTasks()
    
    def run(self, user_query: str) -> str:
        """
        Execute the multi-agent workflow for a user query

        Args:
            user_query: Veterinary question from the user
        
        Returns:
            Final reviewed response
        """
        logger.info(f"Processing query: {user_query}")

        # Initialize agents
        classification_agent = self.agent_manager.classification_agent()
        db_retrieval_agent = self.agent_manager.db_retrieval_agent()
        specialist_agent = self.agent_manager.veterinary_specialist_agent()
        qc_agent = self.agent_manager.quality_control_agent()

        # Create tasks with dependencies
        classification_task = self.task_manager.classification_task(classification_agent, user_query)
        db_retrieval_task = self.task_manager.db_retrieval_task(db_retrieval_agent, context=[classification_task])
        specialist_task = self.task_manager.specialist_response_task(specialist_agent, user_query, context=[classification_task, db_retrieval_task])
        qc_task = self.task_manager.quality_check_task(qc_agent, context=[classification_task, specialist_task])

        # Create and run crew
        crew = Crew(
            agents=[classification_agent, db_retrieval_agent, specialist_agent, qc_agent],
            tasks=[classification_task, db_retrieval_task, specialist_task, qc_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        logger.info("Query processing completed")
        return result
    
# ===================================================
# MAIN EXECUTION (for testing)
# ===================================================

if __name__ == "__main__":
    test_queries = [
        # VETERINARIA - In knowledge base
        "Mi perro comió chocolate hace 1 hora, ¿qué hago?",
        "¿Cuáles son los síntomas del parvovirus?",
        "Perro con vómitos y diarrea con sangre, está muy débil",

        # VETERINARIA - NOT in knowledge base
        "Qué es la leishmaniasis canina",

        # SISTEMA
        "Hola, ¿qué puedes hacer?",
        "Tengo dolor de cabeza",
    ]

    vet_crew = VeterinaryCrew()

    print("\n" + "="*30)
    print("VETERINARY CHATBOT - TESTING")
    print("="*30)

    # Test one query at a time (because of LiteLLM token limitations)
    try:
        response = vet_crew.run("Hola")
        print(f"\nRESPUESTA:\n{response}\n")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")

    
    # Test all queries at the same time (not possible with LiteLLM)
    # for i, query in enumerate(test_queries, 1):
    #     print(f"\n{'='*30}")
    #     print(f"TEST {i}/{len(test_queries)}: {query}")
    #     print(f"{'='*30}\n")

    #     try:
    #         response = vet_crew.run(query)
    #         print(f"\nRESPUESTA:\n{response}\n")
    #     except Exception as e:
    #         logger.error(f"Error: {str(e)}")
    #         print(f"Error: {str(e)}")