import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. Настройки
load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- ПОИСКОВЫЕ ФУНКЦИИ ---
def search_vectors(query):
    vector = model.encode(query).tolist()
    response = supabase.rpc("match_documents", {
        "query_embedding": vector,
        "match_threshold": 0.3, # Порог чуть ниже, чтобы захватить больше контекста
        "match_count": 10       # Берем 10 кандидатов
    }).execute()
    return response.data if response.data else []

def search_keywords(query):
    response = supabase.rpc("kw_match_documents", {
        "query_text": query,
        "match_count": 10       # Берем 10 кандидатов
    }).execute()
    return response.data if response.data else []

def rrf_fusion(semantic_results, keyword_results, k=60):
    fused_scores = {}
    doc_content = {} 

    # Сливаем два списка
    for doc_list in [semantic_results, keyword_results]:
        for rank, doc in enumerate(doc_list):
            doc_id = doc['id']
            doc_content[doc_id] = doc 
            if doc_id not in fused_scores: fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (rank + k)

    # Сортируем
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Возвращаем топ-5 лучших
    final_results = []
    for doc_id, score in sorted_ids[:5]:
        final_results.append(doc_content[doc_id])
    
    return final_results

# --- ГЛАВНАЯ ФУНКЦИЯ RAG ---
def ask_hybrid_bot(question):
    print(f"\n👤 Вопрос: {question}")
    print("SEARCHING... Запускаю гибридный поиск...")
    
    # 1. Параллельный поиск
    vec_res = search_vectors(question)
    kw_res = search_keywords(question)
    
    # 2. RRF Слияние
    top_docs = rrf_fusion(vec_res, kw_res)
    
    if not top_docs:
        print("❌ Ничего не найдено.")
        return

    # 3. Собираем контекст
    context_text = ""
    for doc in top_docs:
        context_text += doc['content'] + "\n---\n"
        
    print("🧠 THINKING... Генерирую ответ...")

    # 4. Запрос к GPT
    response = openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Ты юрист. Отвечай кратко и точно, используя только контекст."},
            {"role": "user", "content": f"Контекст:\n{context_text}\n\nВопрос: {question}"}
        ],
        model="gpt-3.5-turbo",
    )
    
    print("\n" + "="*50)
    print(f"🤖 ОТВЕТ:\n{response.choices[0].message.content}")
    print("="*50)

if __name__ == "__main__":
    while True:
        q = input("\nВведите вопрос (или 'exit'): ")
        if q.lower() == 'exit': break
        ask_hybrid_bot(q)