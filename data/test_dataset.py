from pathlib import Path

import deepdish as dd
from torch.utils.data import Dataset

class TestDataset(Dataset):
    """
    定义测试集
    """

    def __init__(self, mixture_dataset, clean_dataset, limit=None, offset=0):
        """
        构建测试数据集
        Args:
            mixture_dataset (str): 带噪语音数据集
            clean_dataset (str): 纯净语音数据集
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
        """
        mixture_dataset = Path(mixture_dataset)
        clean_dataset = Path(clean_dataset)

        assert mixture_dataset.exists() and clean_dataset.exists(), "测试数据集不存在"

        print(f"Loading mixture dataset {mixture_dataset} ...")
        self.mixture_dataset = dd.io.load("mixture_dataset")
        print(f"Loading clean dataset {clean_dataset} ...")
        self.clean_dataset = dd.io.load("clean_dataset")
        assert len(self.mixture_dataset) == len(self.clean_dataset), \
            "mixture dataset 与 clean dataset 长度不同"

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
        mixture = self.mixture_dataset[name]
        clean = self.clean_dataset[name]

        assert mixture.shape == clean.shape
        return mixture.reshape(1, -1), clean.reshape(1, -1), name
