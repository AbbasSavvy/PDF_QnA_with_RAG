# Paths
ASSETS_DIR = "assets"

# Chunking
CHUNK_SIZE = 768
CHUNK_OVERLAP = 128

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Weaviate
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080
COLLECTION_NAME = "Document"

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2"

# Retrieval
TOP_K = 10
CERTAINTY_THRESHOLD = 0.70
HYBRID_ALPHA = 0.5

# Conversation memory
MAX_HISTORY_TURNS = 3

# Groq
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"