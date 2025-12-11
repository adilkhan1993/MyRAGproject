from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from .config import VECTORSTORE_DIR

def build_vectorstore(chunks):
    print(f"[VECTORSTORE] Создаю Chroma в {VECTORSTORE_DIR}")
    # Используем вашу скачанную модель all-minilm
    embeddings = OllamaEmbeddings(model="all-minilm") 
    vs = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )
    print("[VECTORSTORE] Векторное хранилище создано.")
    return vs

def load_vectorstore():
    print(f"[VECTORSTORE] Загружаю Chroma из {VECTORSTORE_DIR}")
    embeddings = OllamaEmbeddings(model="all-minilm")
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )
    return vs