import pandas as pd

# === 1. Завантаження фінального датасету ===
df = pd.read_csv("datasets/multi_disorders/user_dataset_with_all_labels_and_emotions.csv")

# Перевіряємо структуру
print(f"Розмір датасету: {df.shape}")
print(df.columns)

# === 2. Конвертація дати ===
df["post_created"] = pd.to_datetime(df["post_created"], errors="coerce")

# === 3. Створюємо колонку для супутніх розладів ===
disorders = ["anger_pred", "anxiety_pred", "depression_pred", "narcissistic_pred", "panic_pred"]
df["has_comorbidity"] = df[disorders].sum(axis=1) > 0

# === 4. Фільтр для PTSD ===
df["is_ptsd"] = df["ptsd_pred"] == 1

# === 5. Фільтр для PTSD з супутніми розладами ===
df["ptsd_plus_comorbidity"] = df["is_ptsd"] & df["has_comorbidity"]

# === 6. Фільтр для PTSD із негативним sentiment ===
df["ptsd_negative"] = df["is_ptsd"] & (df["sentiment"] == "negative")

# === 7. Агрегуємо по користувачам ===
user_stats = df.groupby("user_id").agg(
    date_first=("post_created", "min"),
    date_last=("post_created", "max"),
    total_posts=("post_id", "count"),
    ptsd_count=("is_ptsd", "sum"),
    ptsd_plus_comorbidity_count=("ptsd_plus_comorbidity", "sum"),
    ptsd_negative_count=("ptsd_negative", "sum")
).reset_index()

# === 8. Розрахунок відсотків ===
user_stats["ptsd_pct"] = (user_stats["ptsd_count"] / user_stats["total_posts"] * 100).round(2)
user_stats["ptsd_plus_comorbidity_pct"] = (
    user_stats["ptsd_plus_comorbidity_count"] / user_stats["total_posts"] * 100
).round(2)

# === 9. Сортування за кількістю PTSD-постів ===
user_stats = user_stats.sort_values(by="ptsd_count", ascending=False)

# === 10. Збереження результату ===
output_path = "datasets/multi_disorders/user_level_ptsd_analysis.csv"
user_stats.to_csv(output_path, index=False)
print(f"📂 Зведена таблиця збережена: {output_path}")

# === 11. Перевірка перших 10 користувачів ===
print(user_stats.head(10))