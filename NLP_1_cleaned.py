import os
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import evaluate
from peft import LoraConfig, get_peft_model

# ===============================
# Reproducibility
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ===============================
# Load IMDb dataset and tokenizer
# ===============================
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=SEED).select(range(2000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=SEED).select(range(500))

# ===============================
# Metric computation
# ===============================
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# ===============================
# Check for CUDA/fp16 support
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = torch.cuda.is_available()

# ===============================
# Training/Evaluation Helper
# ===============================
def run_trainer(model, training_args, train_dataset, eval_dataset, compute_metrics):
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        eval_results = trainer.evaluate()
        return trainer, eval_results.get('eval_accuracy', None)
    except Exception as e:
        print(f"Error during training/evaluation: {e}")
        return None, None

# ===============================
# Fine-tuning Strategies
# ===============================
results = {}
trainers = {}

print("\n--- Running Basic Standard Fine-Tuning ---")
model_basic_ft = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2).to(device)
training_args_basic = TrainingArguments(
    output_dir="basic_ft_trainer",
    evaluation_strategy="epoch",
    seed=SEED,
    fp16=use_fp16
)
trainer_basic_ft, acc_basic = run_trainer(
    model_basic_ft, training_args_basic, small_train_dataset, small_eval_dataset, compute_metrics
)
results['Basic Standard FT'] = acc_basic
trainers['Basic'] = trainer_basic_ft
print(f"Basic Standard FT Accuracy: {acc_basic}")

print("\n--- Running Tuned Standard Fine-Tuning ---")
model_tuned_ft = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2).to(device)
training_args_tuned = TrainingArguments(
    output_dir="tuned_ft_trainer",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    seed=SEED,
    fp16=use_fp16
)
trainer_tuned_ft, acc_tuned = run_trainer(
    model_tuned_ft, training_args_tuned, small_train_dataset, small_eval_dataset, compute_metrics
)
results['Tuned Standard FT'] = acc_tuned
trainers['Tuned'] = trainer_tuned_ft
print(f"Tuned Standard FT Accuracy: {acc_tuned}")

print("\n--- Running LoRA Fine-Tuning ---")
model_lora = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2).to(device)
# Note: For BERT, target_modules may need to be ["query", "value"] or ["self.query", "self.value"] depending on PEFT version/model. Adjust if you get errors.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)
model_lora = get_peft_model(model_lora, lora_config)
print("Trainable parameters for LoRA model:")
model_lora.print_trainable_parameters()
trainer_lora, acc_lora = run_trainer(
    model_lora, training_args_tuned, small_train_dataset, small_eval_dataset, compute_metrics
)
results['LoRA FT'] = acc_lora
trainers['LoRA'] = trainer_lora
print(f"LoRA FT Accuracy: {acc_lora}")

# ===============================
# Results Comparison & Visualization
# ===============================
print("\n--- Final Comparison of Accuracies ---")
for strategy, accuracy in results.items():
    print(f"{strategy}: {accuracy if accuracy is not None else 'N/A'}")

plt.figure(figsize=(8, 6))
sns.barplot(x=list(results.keys()), y=[v if v is not None else 0 for v in results.values()])
plt.title('Comparison of Fine-Tuning Strategies (Accuracy on Small Test Set)')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--')
plt.show()

# ===============================
# (Optional) Plot Loss Curves
# ===============================
def get_epoch_metrics(trainer, metric_key):
    if trainer is None:
        return []
    history = trainer.state.log_history
    epoch_metrics = {}
    for log_entry in history:
        if 'epoch' in log_entry and metric_key in log_entry:
            epoch = int(log_entry['epoch'])
            epoch_metrics[epoch] = log_entry[metric_key]
    return [v for k, v in sorted(epoch_metrics.items())]

loss_metrics = {k: get_epoch_metrics(v, "loss") for k, v in trainers.items()}
eval_loss_metrics = {k: get_epoch_metrics(v, "eval_loss") for k, v in trainers.items()}

plt.figure(figsize=(10, 6))
for label, values in loss_metrics.items():
    if values:
        plt.plot(range(1, len(values)+1), values, marker='o', label=f'{label} Train Loss')
for label, values in eval_loss_metrics.items():
    if values:
        plt.plot(range(1, len(values)+1), values, linestyle='--', marker='x', label=f'{label} Eval Loss')
plt.title("Training & Evaluation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show() 