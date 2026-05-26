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
from torch.utils.data import DataLoader, Dataset, Subset

from dataset import MultiSliceObjectDataset
from model import SliceAttentionClassifier


ROOT = Path(r"E:\wjz\test1\dataset\dataset_obj")
BASELINE_ROOT = ROOT / "slice_attention_baseline"
DATASET_ROOT = BASELINE_ROOT / "dataset"
ARTIFACT_DIR = BASELINE_ROOT / "artifacts"
MODEL_PATH = BASELINE_ROOT / "slice_attention_model.pth"
DEFAULT_CLASSES = ["chair", "desk", "sofa", "bed", "toilet", "image2d"]
INPUT_MODES = ["multi", "single-gate", "single-gate-black"]

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
LR = 3e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP = 1.0
EMA_ALPHA = 0.35
VAL_RATIO = 0.2
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train slice-attention classifier on gated 3D object slices.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--expected-num-slices", type=int, default=3)
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="multi")
    parser.add_argument("--single-gate-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--min-lr", type=float, default=MIN_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--ema-alpha", type=float, default=EMA_ALPHA)
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
    # 优先保证训练曲线可复现，而不是使用 cuDNN 中最快但可能不确定的卷积算法。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


class SliceInputViewDataset(Dataset):
    """Convert a multi-slice sample into a selected experimental input view."""

    def __init__(self, dataset: Dataset, input_mode: str, single_gate_index: int) -> None:
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
        self.dataset = dataset
        self.input_mode = input_mode
        self.single_gate_index = single_gate_index
        self.samples = dataset.samples
        self.class_names = dataset.class_names
        self.num_slices = self._infer_num_slices()
        if not 0 <= single_gate_index < self.num_slices:
            raise ValueError(
                f"single_gate_index must be in [0, {self.num_slices - 1}], got {single_gate_index}"
            )
        self.effective_num_input_slices = 1 if input_mode == "single-gate" else self.num_slices

    def _infer_num_slices(self) -> int:
        if hasattr(self.dataset, "num_slices"):
            return int(self.dataset.num_slices)
        x, _, _ = self.dataset[0]
        return int(x.shape[0])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y, meta = self.dataset[index]
        meta = dict(meta)
        meta["input_mode"] = self.input_mode
        meta["single_gate_index"] = self.single_gate_index

        if self.input_mode == "multi":
            return x, y, meta

        selected = x[self.single_gate_index : self.single_gate_index + 1]
        if self.input_mode == "single-gate":
            return selected, y, meta

        black_input = torch.zeros_like(x)
        black_input[self.single_gate_index] = x[self.single_gate_index]
        return black_input, y, meta


def stratified_split(dataset: MultiSliceObjectDataset, val_ratio: float, seed: int) -> tuple[Subset, Subset]:
    """按类别分层划分训练集和验证集，避免验证集中某些类别过多或过少。"""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    indices_by_label: dict[int, list[int]] = {}
    for index, sample in enumerate(dataset.samples):
        indices_by_label.setdefault(sample.label, []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in sorted(indices_by_label):
        class_indices = indices_by_label[label][:]
        rng.shuffle(class_indices)
        val_count = int(round(len(class_indices) * val_ratio))
        if len(class_indices) > 1:
            val_count = min(max(val_count, 1), len(class_indices) - 1)

        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def ema(values: list[float], alpha: float) -> list[float]:
    """指数滑动平均，只用于让曲线图更容易观察趋势。"""

    if not values:
        return []
    alpha = min(max(alpha, 0.0), 1.0)
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def plot_curves(history: dict[str, list[float]], out_path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    # 保留原始指标曲线，同时增加 EMA 平滑曲线，方便观察整体趋势。
    train_loss_smooth = ema(history["train_loss"], history["ema_alpha"][0])
    val_loss_smooth = ema(history["val_loss"], history["ema_alpha"][0])
    val_acc_smooth = ema(history["val_acc"], history["ema_alpha"][0])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], color="tab:blue", alpha=0.25, label="train raw")
    plt.plot(epochs, history["val_loss"], color="tab:orange", alpha=0.25, label="val raw")
    plt.plot(epochs, train_loss_smooth, color="tab:blue", linewidth=2.0, label="train EMA")
    plt.plot(epochs, val_loss_smooth, color="tab:orange", linewidth=2.0, label="val EMA")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], color="tab:green", alpha=0.25, label="val acc raw")
    plt.plot(epochs, val_acc_smooth, color="tab:green", linewidth=2.0, label="val acc EMA")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylim(0.0, 1.0)
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


def save_history_csv(history: dict[str, list[float]], out_path: Path) -> None:
    alpha = history["ema_alpha"][0]
    # CSV 同时保存原始值和平滑值，后续重新画图或写论文时可以直接使用。
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "val_acc",
        "train_loss_ema",
        "val_loss_ema",
        "val_acc_ema",
    ]
    rows = []
    train_loss_smooth = ema(history["train_loss"], alpha)
    val_loss_smooth = ema(history["val_loss"], alpha)
    val_acc_smooth = ema(history["val_acc"], alpha)
    for idx, (train_loss, val_loss, val_acc, lr) in enumerate(
        zip(history["train_loss"], history["val_loss"], history["val_acc"], history["lr"]),
        start=1,
    ):
        rows.append(
            {
                "epoch": idx,
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss_ema": train_loss_smooth[idx - 1],
                "val_loss_ema": val_loss_smooth[idx - 1],
                "val_acc_ema": val_acc_smooth[idx - 1],
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    base_dataset = MultiSliceObjectDataset(
        class_dirs=build_class_dirs(args.dataset_root, args.classes),
        transform=transform_gray,
        expected_num_slices=args.expected_num_slices,
    )
    dataset = SliceInputViewDataset(base_dataset, args.input_mode, args.single_gate_index)

    train_set, val_set = stratified_split(dataset, args.val_ratio, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_meta,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_meta,
        pin_memory=device.type == "cuda",
    )

    model = SliceAttentionClassifier(num_classes=len(dataset.class_names)).to(device)
    # label smoothing 能降低模型在小数据集上的过度自信。
    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # 验证集 loss 不使用 smoothing，这样数值仍然对应标准交叉熵。
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 余弦退火会逐渐降低学习率，减少训练后期的曲线震荡。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": [], "ema_alpha": [args.ema_alpha]}
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

            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(x)
            loss = train_criterion(logits, y)
            loss.backward()
            # 梯度裁剪限制偶发的大梯度，避免某个 batch 导致 loss 突然尖峰。
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_total += x.size(0)

        train_loss = running_loss / max(running_total, 1)
        metrics = evaluate(model, val_loader, eval_criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["val_acc"].append(metrics["acc"])
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"val_acc={metrics['acc']:.4f}"
        )

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_rows = metrics["rows"]
            best_cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(range(len(dataset.class_names))))
            torch.save(model.state_dict(), args.model_path)

        # 每个 epoch 的参数更新结束后再调整学习率，和日志中记录的 lr 保持一致。
        scheduler.step()

    plot_curves(history, args.artifact_dir / "training_curves.png")
    save_history_csv(history, args.artifact_dir / "training_history.csv")

    if best_cm is not None:
        plot_confusion(best_cm, dataset.class_names, args.artifact_dir / "best_confusion_matrix.png")

    save_attention_csv(best_rows, args.artifact_dir / "val_attention_weights.csv")

    summary = {
        "num_samples": len(dataset),
        "num_slices_per_object": dataset.num_slices,
        "effective_num_input_slices": dataset.effective_num_input_slices,
        "classes": dataset.class_names,
        "input_mode": args.input_mode,
        "single_gate_index": args.single_gate_index,
        "best_val_acc": best_acc,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_ratio": args.val_ratio,
        "expected_num_slices": args.expected_num_slices,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "ema_alpha": args.ema_alpha,
        "dataset_root": str(args.dataset_root),
        "artifact_dir": str(args.artifact_dir),
        "model_path": str(args.model_path),
    }
    with (args.artifact_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
