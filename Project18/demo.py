import os
from agent import agent_executor, SYSTEM_PROMPT

def main():
    print("=" * 70)
    print("🤖 АВТОНОМНЫЙ ИИ-ИССЛЕДОВАТЕЛЬ (Проект 18)")
    print("=" * 70)
    print("Я готов провести глубокое исследование на любую тему.")
    print("Например: 'Квантовые вычисления' или 'Влияние микропластика на океан'.")
    print("(Для выхода введите 'выход' или 'exit')\n")

    while True:
        # Интерактивный ввод темы от пользователя
        query = input("🔍 Введите тему для исследования: ").strip()
        
        if query.lower() in ['выход', 'exit', 'quit', 'q']:
            print("👋 Завершение работы. До свидания!")
            break
            
        if not query:
            continue

        print(f"\n⏳ Начинаю исследование по теме: '{query}'...")
        print("Агент гуглит информацию, переходит по ссылкам и анализирует текст.")
        print("Это может занять 1-3 минуты, пожалуйста, подождите...\n")

        try:
            # Отправляем системный промпт и запрос пользователя в LangGraph
            inputs = {
                "messages": [
                    ("system", SYSTEM_PROMPT),
                    ("user", f"Проведи исследование на тему: {query}")
                ]
            }
            
            # Запускаем агента (invoke дожидается полного завершения всех шагов)
            response = agent_executor.invoke(inputs)
            
            # Финальный отчет находится в последнем сообщении
            final_report = response["messages"][-1].content
            
            # Генерируем безопасное имя файла (заменяем пробелы, берем первые 20 символов)
            safe_filename = "".join([c if c.isalnum() else "_" for c in query])[:20]
            filename = f"Report_{safe_filename}.md"
            
            # Сохраняем отчет в Markdown-файл
            with open(filename, "w", encoding="utf-8") as f:
                f.write(final_report)
                
            print(f"✅ ИССЛЕДОВАНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            print(f"📄 Отчет сохранен в файл: {filename}\n")
            
            # Выводим небольшое превью в терминал
            print("-" * 30 + " ПРЕВЬЮ ОТЧЕТА " + "-" * 30)
            print(final_report[:500] + "...\n\n[Остальной текст ищите в файле!]")
            print("-" * 75 + "\n")
            
        except Exception as e:
            print(f"❌ Произошла ошибка во время исследования: {e}\n")

if __name__ == "__main__":
    main()