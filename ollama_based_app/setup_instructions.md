# Ollama Chatbot - Setup Instructions

## Prerequisites
- Python 3.8 or higher
- Ollama installed (download from https://ollama.ai)

## Setup Steps

### 1. Create Virtual Environment
```bash
cd "e:\AI_Chilla\AI_CHILLA_2026\Repositories\Streamlit Apps\ollama_based_app"
python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
.venv\Scripts\activate.bat
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Ollama Models
```bash
ollama pull llama2
ollama pull mistral
ollama pull neural-chat
```

### 5. Run Ollama Service
```bash
ollama serve
```
(Keep this terminal window open)

### 6. Run Streamlit App (in another terminal)
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Troubleshooting

- **Connection Error**: Make sure Ollama is running with `ollama serve`
- **No Models**: Download models with `ollama pull <model-name>`
- **Port 11434 in use**: Change the Ollama URL in the sidebar settings
