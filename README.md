# Veterinary AI Assistant

Multi-agent chatbot using CrewAI, FastAPI, and Streamlit.

## Prerequisites

- Python 3.11.5+
- pip

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd vet_chatbot
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv menv
   ```

3. **Activate the virtual environment:**

   - On macOS/Linux:
     ```bash
     source menv/bin/activate
     ```
   - On Windows:
     ```bash
     menv\Scripts\activate
     ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Create environment variables file:**

   ```bash
   # copy .env.example content and paste to .env while creating it
   cp .env.example .env
   ```

   Then edit `.env` and add your API keys:

   ```bash
   # Edit with your preferred editor
   nano .env  # or vim, code, etc.
   ```

6. **Initialize the vector database:**

   ```bash
   python vector_db.py
   ```

7. **Start the app:**

   ```bash
   streamlit run app.py
   ```

   The UI will open in your browser at `http://localhost:8501`

## Project Structure

```
vet_chatbot/
├── menv/              # Virtual environment (git-ignored)
├── vector_db/         # ChromaDB storage (auto-created)
├── vector_db.py       # Database initialization
├── main.py            # Multi-agent implementation
├── app.py             # Streamlit frontend
├── requirements.txt   # Python dependencies
├── .env               # Environment variables (copy from .env.example)
├── .env.example       # Environment variables template
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Files Auto-Generated During Use

- `vector_db/` - Created when initializing the database
- `__pycache__/` - Python bytecode cache

## Deactivating Virtual Environment

When done working on the project:

```bash
deactivate
```

## Troubleshooting

- If you encounter module import errors, ensure the virtual environment is activated
- For "command not found" errors, verify all dependencies are installed
- Check that Python version is 3.11.5 or higher: `python --version`
