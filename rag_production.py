import os
import time
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from cachetools import TTLCache

# 1. ЗАГРУЗКА НАСТРОЕК
load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Читаем конфиг из .env
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "false").lower() == "true"
USE_CACHE = os.environ.get("USE_CACHE", "false").lower() == "true"
TOP_K = int(os.environ.get("RETRIEVAL_K", 10)) # Сколько искать
TOP_N = int(os.environ.get("RERANK_N", 3))     # Сколько оставлять

# 2. ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
print("⏳ Загружаю модели (это может занять время)...")
# Модель для быстрого поиска (Bi-Encoder)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# Модель для точной сортировки (Cross-Encoder) - она умнее, но медленнее
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 3. НАСТРОЙКА КЕША (Храним до 100 ответов, живут CACHE_TTL секунд)
cache = TTLCache(maxsize=100, ttl=int(os.environ.get("CACHE_TTL", 60)))

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def search_vectors(query):
    vector = embed_model.encode(query).tolist()
    response = supabase.rpc("match_documents", {
        "query_embedding": vector,
        "match_threshold": 0.1, # Порог ниже, чтобы набрать кандидатов для сортировки
        "match_count": TOP_K
    }).execute()
    return response.data if response.data else []

def search_keywords(query):
    response = supabase.rpc("kw_match_documents", {
        "query_text": query,
        "match_count": TOP_K
    }).execute()
    return response.data if response.data else []

# Функция RRF из прошлого урока
def rrf_fusion(semantic_results, keyword_results, k=60):
    fused_scores = {}
    doc_content = {} 
    
    for doc_list in [semantic_results, keyword_results]:
        for rank, doc in enumerate(doc_list):
            doc_id = doc['id']
            doc_content[doc_id] = doc 
            if doc_id not in fused_scores: fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (rank + k)
            
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_content[doc_id] for doc_id, score in sorted_ids] # Возвращаем весь список

# --- ГЛАВНАЯ ЛОГИКА (PIPELINE) ---

def ask_smart_bot(question):
    start_time = time.time() # Засекаем время
    print(f"\n👤 Вопрос: {question}")

    # 1. ПРОВЕРКА КЕША
    if USE_CACHE and question in cache:
        print(f"⚡ CACHE HIT! Ответ найден в памяти.")
        print("="*50)
        print(f"🤖 ОТВЕТ:\n{cache[question]}")
        print("="*50)
        print(f"⏱️ Время ответа: {time.time() - start_time:.4f} сек (Мгновенно!)")
        return

    # 2. ПОИСК (RETRIEVAL)
    t1 = time.time()
    vec_res = search_vectors(question)
    kw_res = search_keywords(question)
    # Объединяем через RRF
    candidates = rrf_fusion(vec_res, kw_res)
    print(f"🔍 Найдено кандидатов: {len(candidates)} (за {time.time() - t1:.4f} сек)")

    # 3. ПЕРЕРАНЖИРОВАНИЕ (RERANKING)
    final_docs = candidates
    if RERANK_ENABLED and candidates:
        t2 = time.time()
        print("⚖️  Запускаю Re-ranking (Cross-Encoder)...")
        
        # Готовим пары [Вопрос, Текст] для нейросети
        pairs = [[question, doc['content']] for doc in candidates]
        
        # Нейросеть оценивает, насколько текст отвечает на вопрос
        scores = rerank_model.predict(pairs)
        
        # Прикрепляем оценки к документам и сортируем
        ranked_docs = []
        for i, doc in enumerate(candidates):
            ranked_docs.append({'doc': doc, 'score': scores[i]})
        
        # Сортируем по оценке (от высокой к низкой)
        ranked_docs = sorted(ranked_docs, key=lambda x: x['score'], reverse=True)
        
        # Оставляем только ТОП-N лучших
        final_docs = [item['doc'] for item in ranked_docs[:TOP_N]]
        
        print(f"✅ Re-ranking завершен за {time.time() - t2:.4f} сек.")
        print(f"   Лучший документ (Score: {ranked_docs[0]['score']:.4f}): {final_docs[0]['content'][:50]}...")
    else:
        final_docs = candidates[:TOP_N] # Просто берем первые попавшиеся

    # 4. ГЕНЕРАЦИЯ (GENERATION)
    t3 = time.time()
    print("🧠 Генерирую ответ через GPT...")
    
    context_text = "\n---\n".join([d['content'] for d in final_docs])
    
    response = openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Ты эксперт. Отвечай используя контекст."},
            {"role": "user", "content": f"Контекст:\n{context_text}\n\nВопрос: {question}"}
        ],
        model="gpt-3.5-turbo",
    )
    answer = response.choices[0].message.content
    
    # Сохраняем в кеш
    if USE_CACHE:
        cache[question] = answer

    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"🤖 ОТВЕТ:\n{answer}")
    print("="*50)
    print(f"⏱️ Полное время: {total_time:.4f} сек")
    print(f"📊 Метрики: Поиск={t3-start_time:.2f}s | GPT={total_time-(t3-start_time):.2f}s")

if __name__ == "__main__":
    while True:
        q = input("\nВведите вопрос (или 'exit'): ")
        if q.lower() == 'exit': break
        ask_smart_bot(q)