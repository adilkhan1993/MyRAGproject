import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. Загрузка настроек
load_dotenv()

# Настраиваем Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Настраиваем OpenAI (ChatGPT)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Настраиваем модель для поиска (та же, что и при загрузке)
model = SentenceTransformer("all-MiniLM-L6-v2")

def ask_bot(question):
    print(f"\n🤔 Вы спросили: {question}")
    print("SEARCHING... Ищу информацию в базе...")
    
    # 1. Поиск (Retrieval)
    query_vector = model.encode(question).tolist()
    response = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": 0.3,
        "match_count": 3
    }).execute()
    
    # Собираем контекст
    context_text = ""
    if response.data:
        for doc in response.data:
            context_text += doc['content'] + "\n---\n"
    else:
        context_text = "Информации в базе не найдено."
        
    print("🧠 THINKING... Анализирую и пишу ответ...")

    # 2. Генерация (Generation через GPT-3.5 или GPT-4)
    # Мы посылаем в ChatGPT инструкцию + контекст + вопрос
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Ты юрист-консультант. Отвечай на вопрос, используя ТОЛЬКО предоставленный контекст. Если ответа нет в контексте, так и скажи."
            },
            {
                "role": "user",
                "content": f"Контекст:\n{context_text}\n\nВопрос: {question}"
            }
        ],
        model="gpt-3.5-turbo", # Это дешевая и быстрая модель
    )

    # 3. Вывод ответа
    answer = chat_completion.choices[0].message.content
    print("\n" + "="*50)
    print(f"🤖 ОТВЕТ AI:\n{answer}")
    print("="*50)

if __name__ == "__main__":
    while True:
        q = input("\nВведите вопрос (или 'exit'): ")
        if q.lower() == 'exit':
            break
        ask_bot(q)