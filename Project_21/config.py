# Файл конфигурации для проекта Text-to-SQL (Проект 21)

# 1. Настройки базовой модели
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# 2. Настройки LoRA (QLoRA)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# 3. Настройки обучения (TrainingArguments)
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_STEPS = 100

# 4. Настройки генерации (Inference)
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.1
TOP_P = 0.9
