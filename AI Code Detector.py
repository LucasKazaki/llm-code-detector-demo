#!/usr/bin/env python3
"""
distillation_trainer.py

Implements knowledge distillation from a full CodeBERT teacher to a smaller student model,
using a local CSV of code snippets and labels ('Ai_generated' vs 'Human_written').
Usage (run in PyCharm or CLI):
  python distillation_trainer.py
"""
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW
from tqdm import tqdm


n_labels = 2

# ---- Configuration via input() ----
csv_path = input("Enter path to CSV with 'code' and 'target' columns: ") or r"\replication_package\replication_package\H-AIRosettaMP.csv"
teacher_name = input("Enter teacher model name (e.g. microsoft/codebert-base): ") or "microsoft/codebert-base"
student_dir = input("Enter output directory for student model: ") or "student_model"
epochs = int(input("Number of distillation epochs [3]: ") or 3)
batch_size = int(input("Batch size [8]: ") or 8)
learning_rate = float(input("Learning rate [5e-5]: ") or 5e-5)

# Label mapping and device
label_map = {"Ai_generated": 0, "Human_written": 1}
n_labels = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load data ----
df = pd.read_csv(csv_path)
# Drop rows with missing code or target
df = df.dropna(subset=["code", "target"])
# Map to numeric labels
df['label'] = df['target'].map(label_map)

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.codes = dataframe['code'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.codes)
    def __getitem__(self, idx):
        code = self.codes[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# ---- Prepare tokenizer and dataset ----
tokenizer = AutoTokenizer.from_pretrained(teacher_name)
full_ds = CodeDataset(df, tokenizer)
# Split 80/20 train/test
total = len(full_ds)
train_size = int(0.8 * total)
test_size = total - train_size
train_ds, test_ds = random_split(full_ds, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# ---- Load teacher model (frozen) ----
teacher = AutoModelForSequenceClassification.from_pretrained(
    teacher_name,
    num_labels=n_labels
).to(device)
for param in teacher.parameters():
    param.requires_grad = False
teacher.eval()

# ---- Build student model via config distillation ----n# Load teacher config and modify
config = AutoConfig.from_pretrained(teacher_name)
config.num_labels = n_labels
config.num_hidden_layers = 6
config.hidden_size = 512
config.intermediate_size = 2048
config.num_attention_heads = 8

student = AutoModelForSequenceClassification.from_config(config).to(device)

# ---- Optimizer ----
optimizer = AdamW(student.parameters(), lr=learning_rate)

# ---- Distillation settings ----
T = 2.0    # temperature
alpha = 0.5

# ---- Training and evaluation functions ----
def train_epoch():
    student.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training Batches", leave=False), 1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # Teacher logits
        with torch.no_grad():
            teacher_logits = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits / T
        # Student logits
        student_outputs = student(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        student_logits = student_outputs.logits / T
        # Losses
        loss_ce = F.cross_entropy(student_outputs.logits, labels)
        loss_kd = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        ) * (T * T)
        loss = alpha * loss_kd + (1 - alpha) * loss_ce
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"  [Batch {batch_idx}/{len(train_loader)}] loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    print(f"  -> Average training loss: {avg_loss:.4f}")


def eval_epoch():
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = student(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100
    print(f"  -> Evaluation accuracy: {acc:.2f}%")
    return acc

# ---- Run distillation ----
print(f"Starting distillation for {epochs} epochs...")
best_acc = 0.0
os.makedirs(student_dir, exist_ok=True)
for epoch in tqdm(range(1, epochs+1), desc="Epochs"):  # outer progress
    print(f"\n=== Epoch {epoch}/{epochs} ===")
    train_epoch()
    acc = eval_epoch()
    # Save best model
    if acc > best_acc:
        best_acc = acc
        student.save_pretrained(student_dir)
        tokenizer.save_pretrained(student_dir)
        print(f"  ** New best model saved (acc={best_acc:.2f}%) **")

print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")
