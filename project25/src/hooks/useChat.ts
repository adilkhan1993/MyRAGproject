import { useState } from "react";

export type Message = { role: "user" | "assistant"; content: string };

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Привет! Я твой ИИ-ассистент. Чем могу помочь сегодня?" }
  ]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (prompt: string) => {
    if (!prompt.trim() || isLoading) return;

    setMessages((prev) => [...prev, { role: "user", content: prompt }, { role: "assistant", content: "" }]);
    setIsLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/generate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      while (true && reader) {
        const { done, value } = await reader.read();
        if (done) break;
        setMessages((prev) => {
          const newMsgs = [...prev];
          newMsgs[newMsgs.length - 1].content += decoder.decode(value, { stream: true });
          return newMsgs;
        });
      }
    } catch {
      setMessages((prev) => [...prev.slice(0, -1), { role: "assistant", content: "❌ Ошибка соединения." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return { messages, sendMessage, isLoading };
}