import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

class ModelInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = os.getenv("MODEL_PATH", "./fine_tuned_model")

    def load_model(self):
        """Загрузка модели и токенайзера в память"""
        print(f"Загрузка модели из {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Если видеокарты нет, модель загрузится на процессор
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("✅ Модель успешно загружена!")

    def generate(self, prompt: str) -> str:
        """Генерация ответа по промпту"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Модель не загружена")
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Генерация с базовыми параметрами
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        
        # Декодируем только новый сгенерированный текст
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()

# Создаем глобальный экземпляр (синглтон)
inference_engine = ModelInference()