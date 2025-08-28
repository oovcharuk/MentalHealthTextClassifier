import pandas as pd
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Крок 1: Завантаження даних
data = pd.read_csv('datasets/multi_disorders/combined_dataset.csv')  # ваш датасет

# Перегляд перших кількох рядків датасету
print(data.head())

# Розділення даних на навчальну та валідаційну вибірки
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'].tolist(),
    data['new_label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Крок 2: Створення класу Dataset
class PTSDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Ініціалізація токенізатора та моделі
model_name = "microsoft/deberta-v3-small"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 класи: з ПТСР та без

# Створення DataLoader
train_dataset = PTSDDataset(train_texts, train_labels, tokenizer)
val_dataset = PTSDDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Крок 3: Донавчання моделі
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Функція для навчання моделі
def train_model(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Training loss: {avg_loss:.4f}")

# Запуск навчання
for epoch in range(3):  # наприклад, 3 епохи
    print(f"Epoch {epoch + 1}/{3}")
    train_model(model, train_loader, optimizer)

# Крок 4: Оцінка моделі
def evaluate_model(model, val_loader):
    model.eval()
    total_accuracy = 0
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total_accuracy += (predictions == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = total_accuracy / len(val_loader.dataset)
    print(f"Validation loss: {avg_loss:.4f}, Validation accuracy: {avg_accuracy:.4f}")

# Оцінка моделі
evaluate_model(model, val_loader)

model_save_path = "./trained_models/ptsd/"

# Збереження моделі
model.save_pretrained(model_save_path)

# Збереження токенізатора
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")