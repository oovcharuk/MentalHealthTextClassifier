#Дорозмітка датасету
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

# Шлях до моделі Anger
model_path = './trained_models/anger/best_trained_model'

# Завантаження токенізатора та моделі
tokenizer_anger = DistilBertTokenizer.from_pretrained(model_path)
model_anger = DistilBertForSequenceClassification.from_pretrained(model_path)

# Використання GPU, якщо є
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_anger = model_anger.to(device)
model_anger.eval()
df = pd.read_csv("datasets/multi_disorders/user_dataset_with_ptsd.csv")
print(df.head())
def predict_anger(texts, model, tokenizer, batch_size=16):
    results = []
    model.eval()

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
df["anger_pred"] = predict_anger(df["post_text"].tolist(), model_anger, tokenizer_anger)

# Перевіряємо результат
print(df[["post_text", "ptsd_pred", "anger_pred"]].head(10))
df.to_csv("datasets/multi_disorders/user_dataset_with_ptsd_anger.csv", index=False)
print("✅ Датасет оновлено: datasets/multi_disorders/user_dataset_with_ptsd_anger.csv")