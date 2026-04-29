r"""
Train and evaluate a model on datasets in:
  E:\VIT\Study Related\2nd Year\SEM IV\BDA\CP\SkinSight-main\Dataset

This script supports:
  1) Acne severity classification from filename prefixes (levle0_*.jpg ... levle3_*.jpg)
  2) HAM10000 skin lesion classification from HAM10000_metadata.csv + image parts

Outputs (created under ./runs/):
  - model checkpoint (.pt)
  - metrics.json (accuracy, per-class precision/recall/F1, confusion matrix, TP/TN/FP/FN)
  - report.txt (human-readable summary)

Usage examples:
  python train_dataset.py --task acne --epochs 8
  python train_dataset.py --task ham10000 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from torchvision import models, transforms
from PIL import Image

try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:  # pragma: no cover
    classification_report = None
    confusion_matrix = None


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "Dataset"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class RunPaths:
    run_dir: Path
    checkpoint_path: Path
    metrics_path: Path
    report_path: Path


def make_run_dir(task: str) -> RunPaths:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / f"{task}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        checkpoint_path=run_dir / "model.pt",
        metrics_path=run_dir / "metrics.json",
        report_path=run_dir / "report.txt",
    )


class AcneFilenameDataset(Dataset):
    """
    Acne Dataset 1: images in JPEGImages with names like levle0_123.jpg.
    We parse level 0..3 from prefix.
    """

    def __init__(self, image_paths: list[Path], transform, class_names: list[str]):
        self.image_paths = image_paths
        self.transform = transform
        self.class_names = class_names

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        name = p.name.lower()
        # supports levle0_, level0_ typos
        y = None
        for i in range(4):
            if name.startswith(f"levle{i}_") or name.startswith(f"level{i}_"):
                y = i
                break
        if y is None:
            raise ValueError(f"Cannot infer label from filename: {p.name}")
        return x, y


class Ham10000Dataset(Dataset):
    """
    HAM10000 from metadata CSV + two image folders.
    """

    def __init__(self, rows: list[dict[str, Any]], img_dirs: list[Path], transform, class_to_idx: dict[str, int]):
        self.rows = rows
        self.img_dirs = img_dirs
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, image_id: str) -> Path:
        # images are usually like {image_id}.jpg
        filename = f"{image_id}.jpg"
        for d in self.img_dirs:
            p = d / filename
            if p.exists():
                return p
        raise FileNotFoundError(f"Missing image for id={image_id}")

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_id = row["image_id"]
        dx = row["dx"]
        y = self.class_to_idx[dx]
        p = self._resolve_path(image_id)
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, y


def split_indices(n: int, val_frac: float, test_frac: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx


def subset_from_indices(ds: Dataset, indices: list[int]) -> Dataset:
    return Subset(ds, indices)


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        if torch.is_tensor(y):
            y = y.to(device=device, dtype=torch.long)
        else:
            y = torch.tensor(y, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))
    return {"loss": total_loss / max(1, total), "acc": correct / max(1, total)}


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, list[int], list[int]]:
    model.eval()
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in loader:
        x = x.to(device)
        y_t = list(map(int, y))
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_true.extend(y_t)
        y_pred.extend(preds)
        correct += sum(int(a == b) for a, b in zip(y_t, preds))
        total += len(y_t)
    return correct / max(1, total), y_true, y_pred


def tp_tn_fp_fn(cm: np.ndarray) -> dict[str, Any]:
    # one-vs-rest per class
    per_class = {}
    total = int(cm.sum())
    for i in range(cm.shape[0]):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = total - tp - fp - fn
        per_class[str(i)] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return per_class


def load_ham10000_rows(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    import csv

    meta = dataset_dir / "HAM10000_metadata.csv"
    if not meta.exists():
        raise FileNotFoundError("HAM10000_metadata.csv not found in Skin Dataset 1")
    rows: list[dict[str, Any]] = []
    classes: set[str] = set()
    with meta.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({"image_id": r["image_id"], "dx": r["dx"]})
            classes.add(r["dx"])
    return rows, sorted(classes)


def run_training(
    task: str,
    epochs: int,
    batch_size: int,
    lr: float,
    tune: bool,
    seed: int,
    num_workers: int,
) -> None:
    seed_everything(seed)
    device = get_device()

    run = make_run_dir(task)

    # Transforms
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    if task == "acne":
        ds_dir = DATASET_ROOT / "Acne Dataset 1" / "JPEGImages"
        if not ds_dir.exists():
            raise FileNotFoundError(f"Missing {ds_dir}")
        images = sorted([p for p in ds_dir.glob("*.jpg")])
        if not images:
            raise ValueError("No .jpg images found in Acne Dataset 1/JPEGImages")
        class_names = ["level0", "level1", "level2", "level3"]
        base_ds = AcneFilenameDataset(images, transform=eval_tf, class_names=class_names)
        num_classes = 4
    elif task == "ham10000":
        ds_root = DATASET_ROOT / "Skin Dataset 1"
        rows, classes = load_ham10000_rows(ds_root)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        img_dirs = [ds_root / "HAM10000_images_part_1", ds_root / "HAM10000_images_part_2"]
        base_ds = Ham10000Dataset(rows, img_dirs=img_dirs, transform=eval_tf, class_to_idx=class_to_idx)
        num_classes = len(classes)
        class_names = classes
    else:
        raise ValueError("--task must be one of: acne, ham10000")

    train_idx, val_idx, test_idx = split_indices(len(base_ds), val_frac=0.15, test_frac=0.15, seed=seed)

    # We want augmentations only for training; easiest is to recreate dataset with train_tf for train subset
    if task == "acne":
        ds_dir = DATASET_ROOT / "Acne Dataset 1" / "JPEGImages"
        images = sorted([p for p in ds_dir.glob("*.jpg")])
        train_ds_full = AcneFilenameDataset(images, transform=train_tf, class_names=class_names)
        eval_ds_full = AcneFilenameDataset(images, transform=eval_tf, class_names=class_names)
    else:
        ds_root = DATASET_ROOT / "Skin Dataset 1"
        rows, classes = load_ham10000_rows(ds_root)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        img_dirs = [ds_root / "HAM10000_images_part_1", ds_root / "HAM10000_images_part_2"]
        train_ds_full = Ham10000Dataset(rows, img_dirs=img_dirs, transform=train_tf, class_to_idx=class_to_idx)
        eval_ds_full = Ham10000Dataset(rows, img_dirs=img_dirs, transform=eval_tf, class_to_idx=class_to_idx)

    train_ds = subset_from_indices(train_ds_full, train_idx)
    val_ds = subset_from_indices(eval_ds_full, val_idx)
    test_ds = subset_from_indices(eval_ds_full, test_idx)

    # Windows default multiprocessing spawn can break on local objects; num_workers=0 is safest.
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    def fit_once(lr_value: float, epochs_value: int) -> dict[str, Any]:
        model = build_model(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr_value, weight_decay=1e-4)

        best_val = -1.0
        best_state = None
        history = []
        patience = 3
        patience_left = patience

        print(f"[train] lr={lr_value:.2e} epochs={epochs_value} (device={device})")
        for ep in range(1, epochs_value + 1):
            tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_acc, _, _ = evaluate(model, val_loader, device)
            history.append({"epoch": ep, "train": tr, "val_acc": val_acc})
            print(f"[epoch {ep:02d}] train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} val_acc={val_acc:.4f}")

            if val_acc > best_val + 1e-4:
                best_val = val_acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        test_acc, y_true, y_pred = evaluate(model, test_loader, device)
        print(f"[done] best_val={best_val:.4f} test_acc={test_acc:.4f}")
        return {
            "model": model,
            "val_acc": best_val,
            "test_acc": test_acc,
            "history": history,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    # Simple tuning over LR + epochs
    if tune:
        candidates = [
            {"lr": lr, "epochs": epochs},
            {"lr": lr * 0.5, "epochs": min(epochs + 2, 20)},
            {"lr": lr * 1.5, "epochs": max(epochs - 1, 3)},
        ]
        results = []
        for c in candidates:
            print(f"[tune] trying {c}")
            out = fit_once(c["lr"], c["epochs"])
            results.append({"cfg": c, "val_acc": out["val_acc"], "test_acc": out["test_acc"], "out": out})
        best = sorted(results, key=lambda r: (r["val_acc"], r["test_acc"]), reverse=True)[0]
        final = best["out"]
        chosen = best["cfg"]
        tune_summary = [{"cfg": r["cfg"], "val_acc": r["val_acc"], "test_acc": r["test_acc"]} for r in results]
    else:
        final = fit_once(lr, epochs)
        chosen = {"lr": lr, "epochs": epochs}
        tune_summary = []

    model: nn.Module = final["model"]
    y_true = final["y_true"]
    y_pred = final["y_pred"]
    test_acc = float(final["test_acc"])

    # Metrics
    metrics: dict[str, Any] = {
        "task": task,
        "device": str(device),
        "num_classes": num_classes,
        "class_names": class_names,
        "chosen_hparams": chosen,
        "tuning_runs": tune_summary,
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "test_accuracy": test_acc,
        "history": final["history"],
    }

    if confusion_matrix is not None:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        metrics["confusion_matrix"] = cm.tolist()
        metrics["tp_tn_fp_fn"] = tp_tn_fp_fn(np.asarray(cm))
    if classification_report is not None:
        metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

    # Save model + metrics
    torch.save(
        {
            "task": task,
            "class_names": class_names,
            "state_dict": model.state_dict(),
            "arch": "resnet18",
            "image_size": 224,
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        run.checkpoint_path,
    )
    run.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Human-readable report
    lines = []
    lines.append(f"Task: {task}")
    lines.append(f"Device: {device}")
    lines.append(f"Classes: {class_names}")
    lines.append(f"Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    lines.append(f"Chosen hyperparams: {chosen}")
    lines.append(f"Test accuracy: {test_acc:.4f}")
    if "confusion_matrix" in metrics:
        lines.append("Confusion matrix (rows=true, cols=pred):")
        for row in metrics["confusion_matrix"]:
            lines.append("  " + " ".join(f"{v:5d}" for v in row))
    if "classification_report" in metrics:
        cr = metrics["classification_report"]
        lines.append("Per-class precision/recall/F1:")
        for name in class_names:
            if name in cr:
                lines.append(
                    f"  {name}: P={cr[name]['precision']:.3f} R={cr[name]['recall']:.3f} F1={cr[name]['f1-score']:.3f} support={int(cr[name]['support'])}"
                )
        lines.append(
            f"Macro avg F1: {cr.get('macro avg', {}).get('f1-score', 0):.3f} | Weighted avg F1: {cr.get('weighted avg', {}).get('f1-score', 0):.3f}"
        )
    run.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] Run saved to: {run.run_dir}")
    print(f"[OK] Model: {run.checkpoint_path}")
    print(f"[OK] Metrics: {run.metrics_path}")
    print(f"[OK] Report: {run.report_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["acne", "ham10000"], default="acne")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()
    run_training(
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tune=args.tune,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

