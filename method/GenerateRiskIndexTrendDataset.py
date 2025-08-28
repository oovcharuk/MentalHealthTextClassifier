import pandas as pd
import matplotlib.pyplot as plt

# === 1. Завантажуємо датасет ===
df = pd.read_csv("datasets/multi_disorders/user_dataset_with_all_labels_and_emotions.csv")

# === 2. Конвертуємо дату ===
df["post_created"] = pd.to_datetime(df["post_created"], errors="coerce")

# === 3. Додаємо прапорці для PTSD, коморбідності та негативу ===
disorders = ["anger_pred", "anxiety_pred", "depression_pred", "narcissistic_pred", "panic_pred"]
df["is_ptsd"] = df["ptsd_pred"] == 1
df["has_comorbidity"] = df[disorders].sum(axis=1) > 0
df["ptsd_plus_comorbidity"] = df["is_ptsd"] & df["has_comorbidity"]
df["ptsd_negative"] = df["is_ptsd"] & (df["sentiment"] == "negative")

# === 4. Встановлюємо часовий вікно (наприклад, 30 днів) ===
window_days = 30
df = df.sort_values(by=["user_id", "post_created"])

# === 5. Обчислюємо ковзні метрики по користувачу ===
risk_data = []

for user_id, group in df.groupby("user_id"):
    group = group.set_index("post_created").sort_index()

    # Рухоме вікно
    rolling = group["is_ptsd"].rolling(f"{window_days}D").sum()
    total_posts = group["post_id"].rolling(f"{window_days}D").count()
    ptsd_rate = (rolling / total_posts).fillna(0)

    # Розрахунок boost-факторів
    rolling_comorbidity = group["ptsd_plus_comorbidity"].rolling(f"{window_days}D").sum()
    rolling_negative = group["ptsd_negative"].rolling(f"{window_days}D").sum()

    comorbidity_boost = (rolling_comorbidity / rolling).fillna(0)
    comorbidity_boost = comorbidity_boost.apply(lambda x: 0.2 if x >= 0.3 else 0)

    negative_boost = (rolling_negative / rolling).fillna(0)
    negative_boost = negative_boost.apply(lambda x: 0.3 if x >= 0.5 else 0)

    # Індекс ризику
    risk_index = ptsd_rate + comorbidity_boost + negative_boost
    risk_index = risk_index.clip(0, 1)

    group_result = pd.DataFrame({
        "user_id": user_id,
        "post_created": group.index,
        "risk_index": risk_index
    })
    risk_data.append(group_result)

# Об'єднуємо всі результати
risk_df = pd.concat(risk_data).reset_index(drop=True)

# === 6. Візуалізація тренду PTSD-постів по часу ===
ptsd_trend = df[df["is_ptsd"]].groupby(df["post_created"].dt.to_period("M")).size()
ptsd_trend.index = ptsd_trend.index.to_timestamp()

plt.figure(figsize=(12, 5))
plt.plot(ptsd_trend.index, ptsd_trend.values, marker="o", color="crimson", linewidth=2)
plt.title("📈 Тренд кількості PTSD-постів по місяцях", fontsize=14)
plt.xlabel("Дата")
plt.ylabel("Кількість PTSD постів")
plt.grid()
plt.show()

# === 7. Візуалізація ризику для топ-5 користувачів ===
top_users = df.groupby("user_id")["is_ptsd"].sum().sort_values(ascending=False).head(5).index
plt.figure(figsize=(14, 6))

for user in top_users:
    user_data = risk_df[risk_df["user_id"] == user]
    plt.plot(user_data["post_created"], user_data["risk_index"], label=f"User {user}", linewidth=2)

plt.title(f"📊 Індекс ризикованої поведінки для топ-{len(top_users)} користувачів ({window_days}-денне вікно)")
plt.xlabel("Дата")
plt.ylabel("Risk Index")
plt.legend()
plt.grid()
plt.show()

# === 8. Збереження індексу ризику ===
risk_df.to_csv("datasets/multi_disorders/user_risk_index_trend.csv", index=False)
print("✅ Індекс ризику збережено: datasets/multi_disorders/user_risk_index_trend.csv")