"""Central settings. Values can be overridden from a .env file (see .env.example)."""
import os

from dotenv import load_dotenv

load_dotenv(override=True)

DATA_DIR = os.getenv("DATA_DIR", "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
SQLITE_PATH = os.path.join(DATA_DIR, "metadata.db")

# Existing company folder of PDFs that can be indexed with POST /index-folder
LIBRARY_DIR = os.getenv("LIBRARY_DIR", os.path.join(DATA_DIR, "library"))

# Local embedding model (runs on your own machine, no API cost)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# LLM (Gemini). The key lives ONLY in the .env file, never in the code.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# Leave empty to auto-pick the newest stable Gemini "flash" model for your key
LLM_MODEL = os.getenv("LLM_MODEL", "")

# Passages scoring below this are treated as unrelated, and no LLM call is made.
# Tuned later with the 20-question benchmark.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))
SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid")  # "hybrid" or "semantic"

# How much the model "thinks" before answering. "minimal" makes answers much faster.
# Allowed: minimal, low, medium, high, or empty to use the model's default.
# If the chosen model does not accept it, the app retries without it automatically.
LLM_THINKING = os.getenv("LLM_THINKING", "minimal")
