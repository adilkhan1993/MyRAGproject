import os
import autogen

# 1. ВАШ КЛЮЧ OPENAI
my_key = "sk-dummy-key-for-security"

# 2. НАСТРОЙКА МОДЕЛИ
llm_config = {
    "config_list": [
        {
            "model": "gpt-3.5-turbo",
            "api_key": my_key
        }
    ],
    "temperature": 0.8,
}

print("🤖 Создаем ИИ-агентов...")

# 3. СОЗДАНИЕ АГЕНТОВ
moderator = autogen.AssistantAgent(
    name="Moderator",
    system_message=(
        "Ты модератор дебатов. Следи, чтобы участники не уходили от темы. "
        "Дай слово каждому участнику. После нескольких раундов подведи итог. "
        "Когда дебаты логически завершены, напиши 'TERMINATE'."
    ),
    llm_config=llm_config,
)

optimist = autogen.AssistantAgent(
    name="Techno_Optimist",
    system_message=(
        "Ты техно-оптимист. Ты считаешь, что ИИ помогает программистам "
        "работать быстрее, повышает продуктивность и открывает новые возможности. "
        "Отвечай уверенно, ярко и с энтузиазмом."
    ),
    llm_config=llm_config,
)

skeptic = autogen.AssistantAgent(
    name="Skeptic",
    system_message=(
        "Ты скептик. Ты считаешь, что ИИ несет риски: увольнения, галлюцинации, "
        "ошибки в коде, потерю контроля и проблемы с качеством. "
        "Спорь с оптимистом и приводи контраргументы."
    ),
    llm_config=llm_config,
)

analyst = autogen.AssistantAgent(
    name="Financial_Analyst",
    system_message=(
        "Ты финансовый аналитик. Тебя интересуют деньги, окупаемость, ROI, "
        "стоимость внедрения, выгода для бизнеса и риски."
    ),
    llm_config=llm_config,
)

# 4. НАСТРОЙКА ГРУППОВОГО ЧАТА
print("🏛️ Настраиваем комнату для дебатов...")

groupchat = autogen.GroupChat(
    agents=[moderator, optimist, skeptic, analyst],
    messages=[],
    max_round=12,
 )

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# 5. USER PROXY
user_proxy = autogen.UserProxyAgent(
    name="User",
    system_message="Слушатель дебатов.",
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in str(msg.get("content", "")).upper()
)

# 6. ТЕМА ДЕБАТОВ
topic = "Заменит ли ИИ программистов в ближайшие 10 лет? Начните дебаты со вступительных речей."

print(f"\n🚀 ЗАПУСКАЕМ ДЕБАТЫ!\nТема: {topic}\n" + "=" * 60)

# 7. ЗАПУСК
chat_result = user_proxy.initiate_chat(manager, message=topic)

# 8. СОЗДАНИЕ ПАПКИ ДЛЯ ЛОГОВ
os.makedirs("output", exist_ok=True)

# 9. ПРОСТАЯ ОЦЕНКА УБЕДИТЕЛЬНОСТИ
scores = {
    "Techno_Optimist": 0,
    "Skeptic": 0,
    "Financial_Analyst": 0,
    "Moderator": 0,
}

for message in chat_result.chat_history:
    name = message.get("name", "")
    content = message.get("content", "").lower()

    if name in scores:
        scores[name] += 1

        if "риск" in content or "risk" in content:
            scores[name] += 1
        if "выгода" in content or "roi" in content or "окуп" in content:
            scores[name] += 1
        if "пример" in content or "данные" in content or "статист" in content:
            scores[name] += 1
        if "итог" in content or "вывод" in content:
            scores[name] += 1

winner = max(scores, key=scores.get)

# 10. СОХРАНЕНИЕ ЛОГОВ
print("\n" + "=" * 60)
print("💾 Сохраняем результаты в output/debate_log.md...")

with open("output/debate_log.md", "w", encoding="utf-8") as f:
    f.write("# Лог дебатов\n\n")
    f.write(f"**Тема:** {topic}\n\n")
    f.write("---\n\n")

    for message in chat_result.chat_history:
        name = message.get("name", "Unknown")
        content = message.get("content", "")
        f.write(f"### 🗣️ {name}\n")
        f.write(content + "\n\n")
        f.write("---\n\n")

    f.write("# Анализ дебатов\n\n")
    f.write(f"**Предполагаемый самый убедительный агент:** {winner}\n\n")
    f.write(f"**Scores:** {scores}\n")

print("✅ Готово! Откройте файл output/debate_log.md, чтобы прочитать весь спор.")
print(f"🏆 Предполагаемый победитель дебатов: {winner}")