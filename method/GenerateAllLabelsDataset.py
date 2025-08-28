from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

# Використання GPU, якщо доступний
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Функція для завантаження моделі та токенізатора
def load_model(model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer

# Функція прогнозу для будь-якої моделі
def predict_labels(texts, model, tokenizer, batch_size=16):
    results = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        encoding = tokenizer(batch_texts, return_tensors="pt",
                             padding=True, truncation=True, max_length=128)

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

        results.extend(preds)
    return results

# Завантажуємо датасет із уже доданими PTSD та Anger
df = pd.read_csv("datasets/multi_disorders/user_dataset_with_ptsd_anger.csv")
print(f"✅ Початковий розмір датасету: {df.shape}")
print(df.head(3))

# Шляхи до моделей
model_paths = {
    "anxiety_pred": "./trained_models/anxiety/best_trained_model",
    "depression_pred": "./trained_models/depression/best_trained_model",
    "narcissistic_pred": "./trained_models/narcissistic/best_trained_model",
    "panic_pred": "./trained_models/panic/best_trained_model"
}

# Додаємо колонки для кожної нової моделі
for label_name, path in model_paths.items():
    print(f"\n🔹 Обробка моделі: {label_name}")
    model, tokenizer = load_model(path)
    df[label_name] = predict_labels(df["post_text"].tolist(), model, tokenizer)

# Перевіряємо результат
print("\n✅ Нові колонки додані:")
print(df.head(10))

# Зберігаємо фінальний датасет
output_path = "datasets/multi_disorders/user_dataset_with_all_labels.csv"
df.to_csv(output_path, index=False)
print(f"\n📂 Фінальний датасет збережено: {output_path}")