"use client";

import { useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Привет! Я твой ИИ-ассистент. Чем могу помочь сегодня?" }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false); // Состояние загрузки

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    // 1. Сохраняем сообщение пользователя
    const userPrompt = input;
    const userMessage: Message = { role: "user", content: userPrompt };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // 2. Добавляем пустое сообщение от ИИ, которое будем постепенно заполнять
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      // ВНИМАНИЕ: Замените URL на ваш реальный адрес из Railway!
     const response = await fetch("http://127.0.0.1:8000/generate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: userPrompt }),
      });

      if (!response.ok) throw new Error("Ошибка сети");

      // 3. Читаем потоковый ответ (streaming)
      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");

      if (!reader) return;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        // Дописываем новый кусочек текста к последнему сообщению
        setMessages((prev) => {
          const newMessages = [...prev];
          const lastIndex = newMessages.length - 1;
          newMessages[lastIndex].content += chunk;
          return newMessages;
        });
      }
    } catch (error) {
      console.error(error);
      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1].content = "❌ Ошибка соединения с сервером.";
        return newMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="bg-white shadow px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-800">GenAI Chat</h1>
      </header>

      <main className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((msg, index) => (
          <div key={index} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`p-4 rounded-xl max-w-[80%] shadow-sm ${
              msg.role === "user" ? "bg-blue-600 text-white" : "bg-white border border-gray-200 text-gray-800"
            }`}>
              {msg.content}
            </div>
          </div>
        ))}
      </main>

      <footer className="bg-white p-4 border-t">
        <div className="max-w-4xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Введите ваш промпт..."
            disabled={isLoading}
            className="flex-1 border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 text-black disabled:bg-gray-100"
          />
          <button 
            onClick={handleSend}
            disabled={isLoading}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors shadow-sm disabled:opacity-50"
          >
            Отправить
          </button>
        </div>
      </footer>
    </div>
  );
}