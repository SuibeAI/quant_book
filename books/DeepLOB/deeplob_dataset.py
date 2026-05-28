from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_TRAIN_FILE = "Train_Dst_NoAuction_DecPre_CF_7.txt"
DEFAULT_TEST_FILES = (
    "Test_Dst_NoAuction_DecPre_CF_7.txt",
    "Test_Dst_NoAuction_DecPre_CF_8.txt",
    "Test_Dst_NoAuction_DecPre_CF_9.txt",
)


def prepare_x(raw_data: np.ndarray) -> np.ndarray:
    """提取前 40 行 raw LOB 特征，并转置为 (N, 40)。"""
    return raw_data[:40, :].T


def get_label(raw_data: np.ndarray) -> np.ndarray:
    """提取最后 5 行标签，并转置为 (N, 5)。"""
    return raw_data[-5:, :].T


def build_sequence_samples(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """将逐 event 特征切成长度为 window_size 的序列样本。"""
    num_events, feature_dim = features.shape
    num_samples = num_events - window_size + 1
    if num_samples <= 0:
        raise ValueError(
            f"window_size={window_size} 大于事件数 {num_events}，无法构造样本。"
        )

    sample_x = np.zeros((num_samples, window_size, feature_dim), dtype=np.float32)
    sample_y = labels[window_size - 1 : num_events]

    for idx in range(window_size, num_events + 1):
        sample_x[idx - window_size] = features[idx - window_size : idx, :]

    return sample_x, sample_y


class DeepLOBDataset(Dataset):
    """DeepLOB 的 PyTorch Dataset，单个输入形状为 (1, T, 40)。"""

    def __init__(
        self,
        raw_data: np.ndarray,
        horizon_index: int = 4,
        num_classes: int = 3,
        window_size: int = 100,
    ) -> None:
        self.horizon_index = horizon_index
        self.num_classes = num_classes
        self.window_size = window_size

        features = prepare_x(raw_data)
        labels = get_label(raw_data)
        features, labels = build_sequence_samples(features, labels, window_size)

        labels = labels[:, horizon_index] - 1
        labels = labels.astype(np.int64)

        self.x = torch.from_numpy(features).unsqueeze(1).float()
        self.y = torch.from_numpy(labels).long()
        self.length = len(self.x)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def load_raw_splits(
    data_dir: str | Path,
    train_file: str = DEFAULT_TRAIN_FILE,
    test_files: tuple[str, ...] = DEFAULT_TEST_FILES,
    train_ratio: float = 0.8,
) -> dict[str, np.ndarray]:
    """读取 FI-2010 txt，并按时间切分出 train/val/test 原始矩阵。"""
    data_dir = Path(data_dir)

    train_raw = np.loadtxt(data_dir / train_file)
    split_idx = int(np.floor(train_raw.shape[1] * train_ratio))
    train_raw_split = train_raw[:, :split_idx]
    val_raw_split = train_raw[:, split_idx:]

    test_parts = [np.loadtxt(data_dir / file_name) for file_name in test_files]
    test_raw = np.hstack(test_parts)

    return {
        "train_raw": train_raw,
        "train_raw_split": train_raw_split,
        "val_raw_split": val_raw_split,
        "test_raw": test_raw,
    }


def create_dataloaders(
    data_dir: str | Path,
    window_size: int = 100,
    horizon_index: int = 4,
    num_classes: int = 3,
    batch_size: int = 64,
    train_ratio: float = 0.8,
) -> tuple[dict[str, DeepLOBDataset], dict[str, DataLoader]]:
    """创建 train/val/test 三个 Dataset 和 DataLoader。"""
    raw_splits = load_raw_splits(data_dir=data_dir, train_ratio=train_ratio)

    datasets = {
        "train": DeepLOBDataset(
            raw_splits["train_raw_split"],
            horizon_index=horizon_index,
            num_classes=num_classes,
            window_size=window_size,
        ),
        "val": DeepLOBDataset(
            raw_splits["val_raw_split"],
            horizon_index=horizon_index,
            num_classes=num_classes,
            window_size=window_size,
        ),
        "test": DeepLOBDataset(
            raw_splits["test_raw"],
            horizon_index=horizon_index,
            num_classes=num_classes,
            window_size=window_size,
        ),
    }

    pin_memory = torch.cuda.is_available()
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        ),
    }

    return datasets, dataloaders
