import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# 1. Загрузка настроек
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# 2. Загружаем ту же модель (чтобы "язык" запроса совпадал с базой)
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query):
    print(f"\n🔎 Вопрос: '{query}'")
    
    # А. Превращаем вопрос в вектор
    query_vector = model.encode(query).tolist()
    
    # Б. Отправляем вектор в Supabase (вызываем функцию match_documents)
    response = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": 0.3, # Искать даже отдаленно похожие (0.3 - низкий порог)
        "match_count": 3        # Вернуть топ-3 ответа
    }).execute()
    
    # В. Выводим результат
    if not response.data:
        print("❌ Ничего не найдено (попробуйте переформулировать).")
        return

    print("✅ Найдено в базе:")
    for i, doc in enumerate(response.data):
        print(f"--- Результат #{i+1} (Сходство: {doc['similarity']:.2f}) ---")
        print(f"📄 Текст: {doc['content'][:200]}...") # Показываем первые 200 букв
        print("-" * 50)

if __name__ == "__main__":
    # Тестовые вопросы к Конституции
    search("Кто является источником власти?")
    search("Право на охрану здоровья")