from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

# === Крок 1. Завантаження датасету ===
df = pd.read_csv("datasets/multi_disorders/user_dataset_with_all_labels.csv")
print(f"✅ Початковий розмір датасету: {df.shape}")
print(df.head(3))

# === Крок 2. Завантаження моделі та токенізатора ===
model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# === Крок 3. Список емоцій ===
labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# === Крок 4. Мапінг емоцій на polarity ===
positive_emotions = {
    'admiration','amusement','approval','caring','curiosity','desire','excitement',
    'gratitude','joy','love','optimism','pride','relief'
}
negative_emotions = {
    'anger','annoyance','disappointment','disapproval','disgust','embarrassment',
    'fear','grief','nervousness','remorse','sadness'
}
neutral_emotions = {'neutral','realization','surprise','confusion'}

def map_to_sentiment(emotion):
    if emotion in positive_emotions:
        return "positive"
    elif emotion in negative_emotions:
        return "negative"
    else:
        return "neutral"

# === Крок 5. Функція для визначення головної емоції ===
def predict_emotions(texts, model, tokenizer, batch_size=16):
    dominant_emotions = []
    sentiments = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        encoding = tokenizer(batch_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=256)

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Для кожного тексту знаходимо ключову емоцію
        top_indices = torch.argmax(probs, dim=1).cpu().numpy()
        for idx in top_indices:
            emotion = labels[idx]
            dominant_emotions.append(emotion)
            sentiments.append(map_to_sentiment(emotion))

    return dominant_emotions, sentiments

# === Крок 6. Проганяємо модель ===
df["dominant_emotion"], df["sentiment"] = predict_emotions(
    df["post_text"].tolist(), model, tokenizer
)

# === Крок 7. Зберігаємо оновлений датасет ===
output_path = "datasets/multi_disorders/user_dataset_with_all_labels_and_emotions.csv"
df.to_csv(output_path, index=False)
print(f"\n📂 Фінальний датасет збережено: {output_path}")

# Перевірка перших 10 записів
print(df[["post_text", "dominant_emotion", "sentiment"]].head(10))