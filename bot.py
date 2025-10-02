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
