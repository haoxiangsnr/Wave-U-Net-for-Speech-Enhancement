from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from utils.utils import sample_fixed_length_data_aligned

class TrainDataset(Dataset):
    """
    定义训练集
    """

    def __init__(self, mixture_dataset, clean_dataset, limit=None, offset=0):
        """
        构建训练数据集
        Args:
            mixture_dataset (str): 带噪语音数据集
            clean_dataset (str): 纯净语音数据集
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
        """
        mixture_dataset = Path(mixture_dataset)
        clean_dataset = Path(clean_dataset)

        assert mixture_dataset.exists() and clean_dataset.exists(), "训练数据集不存在"

        print(f"Loading mixture dataset {mixture_dataset.as_posix()} ...")
        self.mixture_dataset:dict = np.load(mixture_dataset.as_posix()).item()
        print(f"Loading clean dataset {clean_dataset.as_posix()} ...")
        self.clean_dataset:dict = np.load(clean_dataset.as_posix()).item()
        assert len(self.mixture_dataset) % len(self.clean_dataset) == 0, \
            "mixture dataset 的长度不是 clean dataset 的整数倍"

        print(f"The len of fully dataset is {len(self.mixture_dataset)}.")
        print(f"The limit is {limit}.")
        print(f"The offset is {offset}.")

        if limit is None:
            limit = len(self.mixture_dataset)

        self.keys = list(self.mixture_dataset.keys())
        self.keys.sort()
        self.keys = self.keys[offset: offset + limit]

        self.length = len(self.keys)
        print(f"Finish, len(finial dataset) == {self.length}.")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        sample_length = 16384

        name = self.keys[item]
        num = name.split("_")[0]
        mixture = self.mixture_dataset[name]
        clean = self.clean_dataset[num]

        assert mixture.shape == clean.shape

        # 定长采样
        mixture, clean = sample_fixed_length_data_aligned(mixture, clean, sample_length)

        return mixture.reshape(1, -1), clean.reshape(1, -1), name
