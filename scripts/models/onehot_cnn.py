"""
CNN Binary Classifier for FASTA Sequences — One-hot only
=========================================================
Classifies sequences as ANCIENT (age > 1000 years) or MODERN (age == 0).

Architecture:
- Raw sequence → one-hot encoded (4 × L) → 1D CNN → binary classification

Expected CSV format:
    ID,AGE,SEQUENCE
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ANCIENT_AGE_THRESHOLD = 2000
SEQ_LENGTH = 15000
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
EARLY_STOPPING_PATIENCE = 6
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 48

TRAIN_CSV = "../data/generated/features/train_features.csv"
VAL_CSV   = "../data/generated/features/val_features.csv"
TEST_CSV  = "../data/generated/features/test_features.csv"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# ENCODING
# ─────────────────────────────────────────────
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def sequence_to_onehot(seq: str, length: int = SEQ_LENGTH) -> np.ndarray:
    seq = seq.upper().strip()
    arr = np.zeros((4, length), dtype=np.float32)
    if len(seq) > length:
        start = (len(seq) - length) // 2
        seq = seq[start: start + length]
    for i, base in enumerate(seq):
        if i >= length:
            break
        idx = BASE_TO_IDX.get(base)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class FASTADataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df[df['AGE'].notna()].reset_index(drop=True) # drop rows with no AGE
        df.loc[df['AGE'] != 0, 'AGE'] = 2026 - df['AGE']
        df.columns = df.columns.str.upper()

        modern_mask  = df["AGE"] == 0
        ancient_mask = df["AGE"] > ANCIENT_AGE_THRESHOLD
        valid_mask   = modern_mask | ancient_mask
        n_dropped = (~valid_mask).sum()
        if n_dropped:
            print(f"[Dataset] Dropping {n_dropped} ambiguous rows "
                f"(0 < age <= {ANCIENT_AGE_THRESHOLD}).")
        df = df[valid_mask].reset_index(drop=True)
        df["LABEL"] = ancient_mask[valid_mask].astype(int).values

        print(f"[Dataset] Encoding {len(df)} sequences ...")
        self.onehots = np.stack(df["SEQUENCE"].apply(sequence_to_onehot).values)
        self.labels  = df["LABEL"].values.astype(np.int64)
        self.ids     = df["ID"].values

        n_mod = (self.labels == 0).sum()
        n_anc = (self.labels == 1).sum()
        print(f"[Dataset] modern={n_mod}, ancient={n_anc} "
            f"(ratio {max(n_mod,n_anc)/max(min(n_mod,n_anc),1):.1f}:1)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.onehots[idx]),   # (4, SEQ_LENGTH)
            torch.tensor(self.labels[idx]),     # scalar
        )


# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int, model_path: str):
        self.patience    = patience
        self.model_path  = model_path
        self.best_auc    = 0.0
        self.counter     = 0
        self.should_stop = False

    def step(self, val_auc: float, model: nn.Module) -> bool:
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.counter  = 0
            torch.save(model.state_dict(), self.model_path)
            print(f"  -> New best saved (val AUC={self.best_auc:.4f})")
        else:
            self.counter += 1
            print(f"  -> No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, pool, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.MaxPool1d(pool),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class MtDNACNN(nn.Module):
    """
    CNN classifier for mtDNA sequences.
    """

    def __init__(self, dropout=DROPOUT):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(4,   32,  kernel=9, pool=4, dropout=0.1),
            ConvBlock(32,  64,  kernel=7, pool=4, dropout=0.1),
            ConvBlock(64,  128, kernel=5, pool=4, dropout=0.2),
            ConvBlock(128, 256, kernel=5, pool=4, dropout=0.2),
            ConvBlock(256, 256, kernel=3, pool=2, dropout=0.2),
            nn.AdaptiveAvgPool1d(16),
        )
        cnn_out_dim = 256 * 16   # 4096

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_out_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

    def forward(self, onehot):
        return self.head(self.cnn(onehot).flatten(1)).squeeze(1)


# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────
def compute_pos_weight(dataset: FASTADataset) -> torch.Tensor:
    n_mod = (dataset.labels == 0).sum()
    n_anc = (dataset.labels == 1).sum()
    return torch.tensor([n_mod / max(n_anc, 1)], dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for onehot, y in loader:
        onehot, y = onehot.to(device), y.to(device).float()
        optimizer.zero_grad()
        logits = model(onehot)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for onehot, y in loader:
        onehot, y = onehot.to(device), y.to(device).float()
        logits = model(onehot)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs > 0.5).astype(int)
        correct += (preds == y.cpu().numpy().astype(int)).sum()
        total += len(y)
        all_probs.extend(probs.tolist())
        all_labels.extend(y.cpu().numpy().astype(int).tolist())
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return total_loss / total, correct / total, auc, all_probs, all_labels


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def plot_training_curves(history, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, keys, title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc"), ("val_auc",)],
        ["Loss", "Accuracy", "Validation AUC-ROC"],
    ):
        for k in keys:
            ax.plot(history[k], label=k.replace("_", " ").title())
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"[Plot] {save_path}")


def plot_confusion_matrix(labels, preds, save_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Modern", "Ancient"],
                yticklabels=["Modern", "Ancient"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[Plot] {save_path}")


def plot_roc_curve(labels, probs, save_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve (Test Set)"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[Plot] {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(train_csv, val_csv, test_csv, output_dir="outputs_onehot"):
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Loading datasets ===")
    train_ds = FASTADataset(train_csv)
    val_ds   = FASTADataset(val_csv)
    test_ds  = FASTADataset(test_csv)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS,
                            pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)

    print(f"\n=== Building model (device: {DEVICE}) ===")
    model = MtDNACNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    pos_weight = compute_pos_weight(train_ds).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(),
                                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=4, factor=0.5
    )

    best_model_path = os.path.join(output_dir, "best_model.pt")
    early_stopping  = EarlyStopping(patience=EARLY_STOPPING_PATIENCE,
                                    model_path=best_model_path)

    print("\n=== Training ===")
    history = {k: [] for k in
            ("train_loss", "val_loss", "train_acc", "val_acc", "val_auc")}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc, val_auc, _, _ = evaluate(
            model, val_loader, criterion, DEVICE
        )
        scheduler.step(val_auc)

        for k, v in zip(history, [train_loss, val_loss, train_acc, val_acc, val_auc]):
            history[k].append(v)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.3f} AUC {val_auc:.4f} | "
            f"lr {lr_now:.2e}"
        )

        if early_stopping.step(val_auc, model):
            print(f"\n=== Early stopping triggered at epoch {epoch} "
                f"(best val AUC={early_stopping.best_auc:.4f}) ===")
            break

    print(f"\n=== Test evaluation (best checkpoint) ===")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    _, test_acc, test_auc, test_probs, test_labels = evaluate(
        model, test_loader, criterion, DEVICE
    )
    test_preds = (np.array(test_probs) > 0.5).astype(int)

    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Test AUC-ROC  : {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=["Modern", "Ancient"]))

    plot_training_curves(history, os.path.join(output_dir, "training_curves.png"))
    plot_confusion_matrix(test_labels, test_preds,
                        os.path.join(output_dir, "confusion_matrix.png"))
    plot_roc_curve(test_labels, test_probs,
                os.path.join(output_dir, "roc_curve.png"))

    print(f"\nAll outputs saved to: {output_dir}/")
    return model


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="CNN-only classifier for ancient/modern mtDNA sequences"
    )
    parser.add_argument("--train",      default=TRAIN_CSV)
    parser.add_argument("--val",        default=VAL_CSV)
    parser.add_argument("--test",       default=TEST_CSV)
    parser.add_argument("--output_dir", default="outputs_onehot")
    args = parser.parse_args()
    main(args.train, args.val, args.test, args.output_dir)
