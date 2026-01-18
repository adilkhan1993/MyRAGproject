import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 1. Загружаем ключи из .env
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    print("❌ Ошибка: Не найдены ключи в файле .env!")
    exit()

# 2. Подключаемся к Supabase
supabase: Client = create_client(url, key)

# 3. Настройки (Файл и Модель)
PDF_PATH = "constitution.pdf"
MODEL_NAME = "all-MiniLM-L6-v2"

def ingest_data():
    print(f"📄 Читаю файл: {PDF_PATH}")
    if not os.path.exists(PDF_PATH):
        print(f"❌ Файл {PDF_PATH} не найден. Проверьте путь!")
        return

    # Загрузка и нарезка
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"🧩 Нарезано на {len(chunks)} частей.")

    # Векторизация
    print("🧠 Генерирую векторы...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Готовим данные для отправки
    data_to_upload = []
    for chunk in chunks:
        text = chunk.page_content
        vector = model.encode(text).tolist() # Превращаем в список [0.1, -0.5...]
        
        data_to_upload.append({
            "content": text,
            "metadata": chunk.metadata,
            "embedding": vector
        })

    # Отправка в Supabase
    print("☁️ Загружаю в облако Supabase...")
    response = supabase.table("documents").insert(data_to_upload).execute()
    
    print("✅ Успешно! Данные теперь в облаке.")

if __name__ == "__main__":
    ingest_data()