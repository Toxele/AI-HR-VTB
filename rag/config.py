# config.py
# Настройки проекта
LM_ADDRESS = 'http://localhost'
LM_PORT = '1234'
MODEL_ID = 'qwen1.5-1.8b-chat'

#Embedding models
TEXT_EMBEDDING_MODEL = 'BAAI/bge-m3'

# FAISS
FAISS_INDEX_PATH = 'data/faiss_index.index'
FAISS_N_PROBE = 15
# Count of relevant docs
TOTAL_DOCS_COUNT = 7
ALPHA = 0.5