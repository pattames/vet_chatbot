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
