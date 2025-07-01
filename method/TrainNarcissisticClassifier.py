# Check versions
import transformers
import torch
import datasets
print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Datasets version: {datasets.__version__}")

# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score, matthews_corrcoef
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import seaborn as sns

# Disable W&B
os.environ["WANDB_DISABLED"] = "true"

# Load dataset
data = pd.read_csv('datasets/multi_disorders/NarcissisticDisorder.csv')

# Split into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Print sample counts
print(f"Number of samples in training set: {len(train_data)}")
print(f"Number of samples in test set: {len(test_data)}")
print(f"Class distribution in training set:\n{train_data['label'].value_counts()}")
print(f"Class distribution in test set:\n{test_data['label'].value_counts()}")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['Text'], truncation=True, padding='max_length', max_length=128)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove text column
train_dataset = train_dataset.remove_columns(['Text'])
test_dataset = test_dataset.remove_columns(['Text'])

# Format labels
def format_labels(examples):
    examples['label'] = [1 if label == 'Narcissistic Disorder' else 0 for label in examples['label']]
    return examples

train_dataset = train_dataset.map(format_labels, batched=True)
test_dataset = test_dataset.map(format_labels, batched=True)

# Set format for PyTorch
train_dataset.set_format('torch')
test_dataset.set_format('torch')

# Configure model with increased Dropout
config = DistilBertConfig.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2,
    hidden_dropout_prob=0.3,  # Increased Dropout for hidden layers
    attention_probs_dropout_prob=0.3  # Increased Dropout for attention
)

# Load model with configuration
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    config=config
)

# Lists to store metrics history per epoch
train_losses = []
eval_losses = []
eval_accuracies = []

# Custom Trainer for logging metrics per epoch
class CustomTrainer(Trainer):
    def log(self, logs: dict, *args, **kwargs):
        super().log(logs)
        # Log only at the end of an epoch
        if 'epoch' in logs and logs['epoch'].is_integer():
            if 'loss' in logs:
                train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                eval_losses.append(logs['eval_loss'])
            if 'eval_accuracy' in logs:
                eval_accuracies.append(logs['eval_accuracy'])

# Function to compute Dice Loss
def dice_loss(preds, true_labels, smooth=1e-6):
    # Convert predictions to binary (using threshold 0.5)
    probs = torch.nn.functional.softmax(torch.tensor(preds), dim=-1).numpy()[:, 1]
    pred_labels = (probs >= 0.5).astype(int)
    true_labels = np.array(true_labels)

    # Compute intersection and union
    intersection = np.sum(pred_labels * true_labels)
    pred_sum = np.sum(pred_labels)
    true_sum = np.sum(true_labels)

    # Dice coefficient
    dice_coeff = (2.0 * intersection + smooth) / (pred_sum + true_sum + smooth)

    # Dice Loss = 1 - Dice Coefficient
    return 1.0 - dice_coeff

# Function to compute Focal Loss
def focal_loss(preds, true_labels, gamma=2.0, alpha=0.25):
    probs = torch.nn.functional.softmax(torch.tensor(preds), dim=-1).numpy()
    true_labels = np.array(true_labels)
    ce_loss = -np.log(probs[np.arange(len(true_labels)), true_labels])
    pt = np.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return np.mean(focal_loss)

# Function to compute extended metrics
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    probs = torch.nn.functional.softmax(torch.tensor(p.predictions), dim=-1).numpy()
    true_labels = p.label_ids

    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')
    auc_score = roc_auc_score(true_labels, probs[:, 1])
    balanced_acc = balanced_accuracy_score(true_labels, preds)
    mcc = matthews_corrcoef(true_labels, preds)

    # Compute confusion matrix for specificity
    cm = confusion_matrix(true_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Compute Focal Loss
    focal_loss_value = focal_loss(p.predictions, true_labels, gamma=2.0, alpha=0.25)

    # Compute Dice Loss
    dice_loss_value = dice_loss(p.predictions, true_labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'balanced_accuracy': balanced_acc,
        'mcc': mcc,
        'specificity': specificity,
        'focal_loss': focal_loss_value,
        'dice_loss': dice_loss_value  # New metric
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./trained_models/narcissistic/results',
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    weight_decay=0.1,
    logging_dir='./logs',
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    save_total_limit=1,
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    warmup_steps=24,
)

# Initialize CustomTrainer with Early Stopping
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Train model
trainer.train()

# Evaluate on training set
train_results = trainer.evaluate(train_dataset)
print("\nEvaluation results on training set:")
for key, value in train_results.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)
print("\nEvaluation results on test set:")
for key, value in test_results.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

# Plot metrics per epoch
epochs = list(range(1, len(eval_losses) + 1))  # Epoch numbers (1, 2, 3, ...)

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(epochs[:len(train_losses)], train_losses, label='Training Loss', marker='o')
plt.plot(epochs[:len(eval_losses)], eval_losses, label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.xticks(epochs)  # Show integer epoch numbers
plt.show()

# Accuracy plot
plt.figure(figsize=(10, 5))
plt.plot(epochs[:len(eval_accuracies)], eval_accuracies, label='Validation Accuracy', marker='x', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.xticks(epochs)  # Show integer epoch numbers
plt.show()

# Detailed classification report and confusion matrix
predictions = trainer.predict(test_dataset)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
true_labels = predictions.label_ids
pred_labels = predictions.predictions.argmax(-1)

# ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(true_labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Narcissistic Disorder Classification')
plt.legend(loc='lower right')
plt.show()

# Optimal threshold (Youden’s J)
youden_index = tpr - fpr
optimal_threshold_index = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_threshold_index]
print(f"\nOptimal threshold (Youden’s J): {optimal_threshold:.4f}")

# Classification with optimal threshold
final_predictions = (probs >= optimal_threshold).astype(int)

# Classification report
print("\nClassification report on test set (with optimal threshold):")
print(classification_report(true_labels, final_predictions))

# Confusion matrix
cm = confusion_matrix(true_labels, final_predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Narcissistic', 'Narcissistic'], yticklabels=['Non-Narcissistic', 'Narcissistic'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on Test Set')
plt.show()

# Save the best model
model.save_pretrained('./trained_models/narcissistic/best_trained_model')
tokenizer.save_pretrained('./trained_models/narcissistic/best_trained_model')