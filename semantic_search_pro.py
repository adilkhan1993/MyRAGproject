import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- НАСТРОЙКИ ---
COLLECTION_NAME = "kaspi_report"
MODEL_NAME = "all-MiniLM-L6-v2"
# Ищем файл рядом со скриптом. Если у вас он в папке data, поменяйте на "data/docs/..."
PDF_PATH = "data/docs/sample.pdf"

class VectorSearchEngine:
    def __init__(self):
        print("⏳ Загружаю нейросеть (если запускаете первый раз, это займет минуту)...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.vector_size = 384
        
        # Запускаем базу в памяти
        self.client = QdrantClient(":memory:")
        
        # Проверяем, есть ли коллекция, если нет — создаем
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def process_pdf(self, file_path: str):
        print(f"📄 Пробую открыть файл: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"⚠️ Файл не найден! Создаю тестовые данные, чтобы показать, как это работает...")
            # Создаем фейковые данные, если PDF нет
            texts = [
                "Kaspi.kz showed strong results in 3Q 2025 with 20% revenue growth.",
                "Hepsiburada acquisition in Türkiye helps international expansion.",
                "Net income increased by 12% year-over-year.",
                "The supply of smartphones remains subject to temporary disruption in Kazakhstan."
            ]
            chunks = [{"text": t, "page": 1} for t in texts]
        else:
            # Если файл есть — читаем его
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(docs)
            chunks = [{"text": d.page_content, "page": d.metadata.get("page", 0)+1} for d in split_docs]
            print(f"🧩 Документ разбит на {len(chunks)} фрагментов.")

        # Векторизация
        print("🚀 Превращаю текст в векторы...")
        points = []
        for i, item in enumerate(chunks):
            text = item["text"]
            vector = self.model.encode(text).tolist()
            
            points.append(PointStruct(
                id=i,
                vector=vector,
                payload={"text": text, "page": item["page"]}
            ))
            
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Готово! В базе {len(points)} векторов.")

    def search(self, query: str):
        if not query: return
        print(f"\n🔎 Ищу: '{query}'")
        query_vector = self.model.encode(query).tolist()
        
        # ИСПРАВЛЕНИЕ: Используем query_points вместо search
        hits = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3,
            with_payload=True
        ).points
        
        print("=" * 50)
        for hit in hits:
            print(f"🎯 Точность: {hit.score:.3f} | Стр. {hit.payload['page']}")
            print(f"📄 ...{hit.payload['text'][:200].replace(chr(10), ' ')}...")
            print("-" * 50)

if __name__ == "__main__":
    app = VectorSearchEngine()
    app.process_pdf(PDF_PATH)
    
    print("\n💡 Теперь можно задавать вопросы (на английском или русском).")
    while True:
        q = input("\nВаш вопрос (или 'q' для выхода): ")
        if q.lower() in ['q', 'exit']: break
        app.search(q)