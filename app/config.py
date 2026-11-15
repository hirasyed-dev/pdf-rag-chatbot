"""Central settings. Values can be overridden from a .env file (see .env.example)."""
import os

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
SQLITE_PATH = os.path.join(DATA_DIR, "metadata.db")

# Existing company folder of PDFs that can be indexed with POST /index-folder
LIBRARY_DIR = os.getenv("LIBRARY_DIR", os.path.join(DATA_DIR, "library"))

# Local embedding model (runs on your own machine, no API cost)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
