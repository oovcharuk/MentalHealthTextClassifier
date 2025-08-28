#Дорозмітка моделлю
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

# Шлях до збереженої моделі
model_path = "./trained_models/ptsd/"

# Завантаження моделі та токенізатора
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
model = DebertaV2ForSequenceClassification.from_pretrained(model_path)

# Використання GPU, якщо є
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
# Завантаження початкового датасету
df = pd.read_csv("datasets/multi_disorders/Mental-Health-Twitter.csv")

# Перевірка структури
print(df.head())
def predict_ptsd(texts, model, tokenizer, batch_size=16):
    results = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        encoding = tokenizer(batch_texts, return_tensors="pt",
                             padding=True, truncation=True, max_length=512)

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

        results.extend(preds)
    return results
# Прогнозуємо ймовірність ПТСР для кожного тексту
df["ptsd_pred"] = predict_ptsd(df["post_text"].tolist(), model, tokenizer)

# Перевіряємо результат
print(df[["post_text", "ptsd_pred"]].head(10))
df.to_csv("datasets/multi_disorders/user_dataset_with_ptsd.csv", index=False)
print("✅ Датасет успішно дорозмічено та збережено: combined_dataset_with_ptsd.csv")