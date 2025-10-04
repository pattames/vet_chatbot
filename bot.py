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

# Load environment vairables from .env
load_dotenv()

# Load Groq credentials
api_key = os.getenv("GROQ_API_KEY")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Veterinary AI Assistant")

# Initialize app state for storing recent queries and clarification attempts
if not hasattr(app.state, "recent_queries"):
    app.state.recent_queries = {} # Stores last 5 queries per session
if not hasattr(app.state, "clarification_attempts"):
    app.state.clarification_attempts = {} # Track clarification attempts per session

# Define LLM
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.3, # Lower for medical accuracy
    max_completion_tokens=1024,
    api_key=api_key
    )

# Define FastAPI request models (structure of data the API expects to receive)
class QueryRequest(BaseModel):
    query: str # Required field
    session_id: str = "default_session" # Tracks which user/conversation the query belongs to (if not provided, automatically uses "default session")