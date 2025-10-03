import os
from typing import List, Dict
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from vector_db import query_diseases
import logging