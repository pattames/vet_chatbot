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
