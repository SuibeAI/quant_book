import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from deeplob_dataset import create_dataloaders
from deeplob_model import DeepLOBModel


def evaluate(model, data_loader, criterion, device):
    """在给定 data_loader 上评估 loss 和 accuracy。"""
    model.eval()
    loss_values = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, dtype=torch.float, non_blocking=True)
            targets = targets.to(device, dtype=torch.int64, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_values.append(loss.item())
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    mean_loss = float(np.mean(loss_values)) if loss_values else 0.0
    accuracy = float(correct / total) if total > 0 else 0.0
    return mean_loss, accuracy


def collect_predictions(model, data_loader, device):
    """收集测试集上的全部真实标签与预测标签。"""
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, dtype=torch.float, non_blocking=True)
            targets = targets.to(device, dtype=torch.int64, non_blocking=True)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    return all_targets, all_predictions


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch，返回 train_loss 和 train_acc。"""
    model.train()
    loss_values = []
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device, dtype=torch.float, non_blocking=True)
        targets = targets.to(device, dtype=torch.int64, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    train_loss = float(np.mean(loss_values)) if loss_values else 0.0
    train_acc = float(correct / total) if total > 0 else 0.0
    return train_loss, train_acc


def fit(model, criterion, optimizer, train_loader, val_loader, epochs, device, checkpoint_path):
    """完整训练流程：每个 epoch 记录 train/val 的 loss 与 acc。"""
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = np.inf
    best_val_epoch = -1

    for epoch in tqdm(range(epochs)):
        epoch_start = datetime.now()

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": model.num_classes,
                },
                checkpoint_path,
            )
            best_val_loss = val_loss
            best_val_epoch = epoch
            print(f"model saved to {checkpoint_path}")

        duration = datetime.now() - epoch_start
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
            f"Duration: {duration}, "
            f"Best Val Epoch: {best_val_epoch}"
        )

    return history


def plot_training_curves(history, output_path):
    """绘制训练过程中的 loss/acc 曲线，并保存到文件。"""
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train acc")
    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLOB on FI-2010.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--horizon-index", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(__file__).resolve().parent / "best_val_model_pytorch.pth",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path(__file__).resolve().parent / "training_history.json",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path(__file__).resolve().parent / "training_curves.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets, dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        window_size=args.window_size,
        horizon_index=args.horizon_index,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
    )

    print("样本张量形状：")
    print("train_dataset.x =", datasets["train"].x.shape)
    print("train_dataset.y =", datasets["train"].y.shape)
    print("val_dataset.x   =", datasets["val"].x.shape)
    print("val_dataset.y   =", datasets["val"].y.shape)
    print("test_dataset.x  =", datasets["test"].x.shape)
    print("test_dataset.y  =", datasets["test"].y.shape)

    model = DeepLOBModel(num_classes=datasets["train"].num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = fit(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        epochs=args.epochs,
        device=device,
        checkpoint_path=args.checkpoint_path,
    )

    history_json = json.dumps(history, ensure_ascii=False, indent=2)
    args.history_path.write_text(history_json, encoding="utf-8")
    print(f"已保存 {args.history_path}")

    plot_training_curves(history, args.plot_path)
    print(f"已保存 {args.plot_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    best_model = DeepLOBModel(num_classes=checkpoint["num_classes"]).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(
        model=best_model,
        data_loader=dataloaders["test"],
        criterion=criterion,
        device=device,
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc:.4f}")

    all_targets, all_predictions = collect_predictions(
        model=best_model,
        data_loader=dataloaders["test"],
        device=device,
    )
    print("accuracy_score:", accuracy_score(all_targets, all_predictions))
    print(classification_report(all_targets, all_predictions, digits=4))


if __name__ == "__main__":
    main()
