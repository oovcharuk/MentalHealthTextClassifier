import pandas as pd
from sklearn.model_selection import train_test_split

# Завантаження даних
train_data = pd.read_csv('datasets/multi_disorders/train.csv')
depression_data = pd.read_csv('datasets/multi_disorders/depression_dataset_reddit_cleaned.csv')

# Фільтрація записів для цільового класу
narcissistic_disorder_data = train_data[train_data['label'] == 'Anxiety Disorder']

# Розрахунок необхідної кількості записів для кожного джерела (20% від кількості цільових записів)
target_count = len(narcissistic_disorder_data)
sample_size_per_source = target_count // 5  # 20% від цільового класу

# Фільтрація записів для нецільових класів
non_target_classes = ['Anxiety Disorder', 'Depression', 'Panic Disorder', 'Anger/ Intermittent Explosive Disorder']

non_target_samples = []
for label in non_target_classes:
    class_data = train_data[train_data['label'] == label]
    class_sample = class_data.sample(n=sample_size_per_source, random_state=42)
    class_sample['label'] = 'non-target'
    non_target_samples.append(class_sample)

# Вибір 20% записів з другого датасету, де is_depression = 0
non_depression_data = depression_data[depression_data['is_depression'] == 0]
non_depression_sample = non_depression_data.sample(n=sample_size_per_source, random_state=42)
non_depression_sample = non_depression_sample[['clean_text']]
non_depression_sample['label'] = 'non-target'
non_depression_sample = non_depression_sample.rename(columns={'clean_text': 'Text'})

# Об'єднання всіх вибраних вибірок у один датасет
combined_data = pd.concat([narcissistic_disorder_data] + non_target_samples + [non_depression_sample])

# Перемішування записів для рівномірного розподілу
final_dataset = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Видалення непотрібної колонки Unnamed: 0 (якщо вона існує)
if 'Unnamed: 0' in final_dataset.columns:
    final_dataset = final_dataset.drop(columns=['Unnamed: 0'])

# Збереження результатів у новий файл
final_dataset.to_csv('datasets/multi_disorders/AnxietyDisorder.csv', index=False)

print(f"Розмір кінцевого датасету: {final_dataset.shape}")
print(final_dataset['label'].value_counts())