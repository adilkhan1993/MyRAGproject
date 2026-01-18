import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# 1. Настройки
load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- ФУНКЦИЯ 1: Векторный поиск (По смыслу) ---
def search_vectors(query):
    vector = model.encode(query).tolist()
    response = supabase.rpc("match_documents", {
        "query_embedding": vector,
        "match_threshold": 0.3,
        "match_count": 5
    }).execute()
    return response.data if response.data else []

# --- ФУНКЦИЯ 2: Ключевой поиск (По словам) ---
def search_keywords(query):
    response = supabase.rpc("kw_match_documents", {
        "query_text": query,
        "match_count": 5
    }).execute()
    return response.data if response.data else []

# --- ФУНКЦИЯ 3: RRF (Слияние результатов) ---
def rrf_fusion(semantic_results, keyword_results, k=60):
    fused_scores = {}
    doc_content = {} 

    # Обрабатываем векторные результаты
    for rank, doc in enumerate(semantic_results):
        doc_id = doc['id']
        doc_content[doc_id] = doc 
        if doc_id not in fused_scores: fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (rank + k)

    # Обрабатываем ключевые результаты
    for rank, doc in enumerate(keyword_results):
        doc_id = doc['id']
        doc_content[doc_id] = doc
        if doc_id not in fused_scores: fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (rank + k)

    # Сортировка
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for doc_id, score in sorted_ids:
        final_results.append(doc_content[doc_id])
    
    return final_results

# --- ЗАПУСК ---
if __name__ == "__main__":
    # Тест: ищем точное совпадение "Статья 10"
    query = "Статья 10" 
    
    print(f"🔎 Запрос: '{query}'")
    
    print("\n--- 🧠 Векторный поиск (Топ-2) ---")
    vec_res = search_vectors(query)
    for doc in vec_res[:2]: print(f"- {doc['content'][:80]}...")

    print("\n--- 🔑 Ключевой поиск (Топ-2) ---")
    kw_res = search_keywords(query)
    for doc in kw_res[:2]: print(f"- {doc['content'][:80]}...")

    print("\n--- 🚀 ГИБРИДНЫЙ РЕЗУЛЬТАТ (RRF) ---")
    hybrid_res = rrf_fusion(vec_res, kw_res)
    for i, doc in enumerate(hybrid_res[:3]):
        print(f"#{i+1}: {doc['content'][:100]}...")