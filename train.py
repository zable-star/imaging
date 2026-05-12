from __future__ import annotations

import csv
import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split

from dataset import MultiSliceObjectDataset
from model import SliceAttentionClassifier


ROOT = Path(r"E:\wjz\test1\dataset\dataset_obj")
BASELINE_ROOT = ROOT / "slice_attention_baseline"
DATASET_ROOT = BASELINE_ROOT / "dataset"
ARTIFACT_DIR = BASELINE_ROOT / "artifacts"
MODEL_PATH = BASELINE_ROOT / "slice_attention_model.pth"
DEFAULT_CLASSES = ["chair", "desk", "sofa", "bed", "toilet"]

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.2
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train slice-attention classifier on gated 3D object slices.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--expected-num-slices", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def build_class_dirs(dataset_root: Path, class_names: list[str]) -> dict[str, Path]:
    class_dirs = {class_name: dataset_root / class_name for class_name in class_names}
    missing = [str(path) for path in class_dirs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing class slice directories:\n" + "\n".join(missing))
    return class_dirs


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize_tensor_img(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == (IMAGE_SIZE, IMAGE_SIZE):
        return x
    return torch.nn.functional.interpolate(
        x.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    ).squeeze(0)


def transform_gray(img):
    from dataset import pil_to_tensor_gray

    return resize_tensor_img(pil_to_tensor_gray(img))


def collate_meta(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), list(metas)


def plot_curves(history: dict[str, list[float]], out_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="val acc")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confusion(cm, class_names, out_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    rows = []

    for x, y, metas in loader:
        x = x.to(device)
        y = y.to(device)
        logits, attn_weights, _ = model(x)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        correct += (preds == y).sum().item()

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        for meta, prob, pred, gt, attn in zip(metas, probs.cpu(), preds.cpu(), y.cpu(), attn_weights.cpu()):
            row = {
                "sample_id": meta["sample_id"],
                "class_name": meta["class_name"],
                "pred": int(pred.item()),
                "gt": int(gt.item()),
            }
            for idx, value in enumerate(attn.tolist()):
                row[f"attn_gate_{idx}"] = float(value)
            for idx, value in enumerate(prob.tolist()):
                row[f"prob_class_{idx}"] = float(value)
            rows.append(row)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "y_true": y_true,
        "y_pred": y_pred,
        "rows": rows,
    }


def save_attention_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = MultiSliceObjectDataset(
        class_dirs=build_class_dirs(args.dataset_root, args.classes),
        transform=transform_gray,
        expected_num_slices=args.expected_num_slices,
    )

    train_len = int(len(dataset) * (1 - args.val_ratio))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_meta)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_meta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SliceAttentionClassifier(num_classes=len(dataset.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_rows = []
    best_cm = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_total += x.size(0)

        train_loss = running_loss / max(running_total, 1)
        metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["val_acc"].append(metrics["acc"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"val_acc={metrics['acc']:.4f}"
        )

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_rows = metrics["rows"]
            best_cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(range(len(dataset.class_names))))
            torch.save(model.state_dict(), args.model_path)

    plot_curves(history, args.artifact_dir / "training_curves.png")

    if best_cm is not None:
        plot_confusion(best_cm, dataset.class_names, args.artifact_dir / "best_confusion_matrix.png")

    save_attention_csv(best_rows, args.artifact_dir / "val_attention_weights.csv")

    summary = {
        "num_samples": len(dataset),
        "num_slices_per_object": dataset.num_slices,
        "classes": dataset.class_names,
        "best_val_acc": best_acc,
        "dataset_root": str(args.dataset_root),
        "model_path": str(args.model_path),
    }
    with (args.artifact_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
