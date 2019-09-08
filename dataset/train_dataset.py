import os

import joblib
from torch.utils.data import Dataset

from utils.utils import sample_fixed_length_data_aligned, apply_mean_std

class TrainDataset(Dataset):
    """
    定义训练集
    """

    def __init__(self, mixture_dataset, clean_dataset, limit=None, offset=0, apply_normalization=False):
        """
        构建训练数据集
        Args:
            mixture_dataset (str): 带噪语音数据集
            clean_dataset (str): 纯净语音数据集
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
            apply_normalization (bool): 是否规范化（减均值，除标准差）
        """
        assert os.path.exists(mixture_dataset) and os.path.exists(clean_dataset), "训练数据集不存在"

        print(f"Loading mixture dataset {mixture_dataset} ...")
        mixture_dataset: dict  = joblib.load(mixture_dataset)
        print(f"Loading clean dataset {clean_dataset} ...")
        clean_dataset: dict = joblib.load(clean_dataset)
        assert len(mixture_dataset) % len(clean_dataset) == 0, \
            "mixture dataset 的长度不是 clean dataset 的整数倍"

        mixture_dataset_keys = list(mixture_dataset.keys())
        mixture_dataset_keys = sorted(mixture_dataset_keys)

        # Limit
        if limit and limit <= len(mixture_dataset_keys):
            self.length = limit
        else:
            self.length = len(mixture_dataset_keys)

        # Offset
        if offset:
            mixture_dataset_keys = mixture_dataset_keys[offset: offset + self.length]
            self.length = len(mixture_dataset_keys)

        self.mixture_dataset = mixture_dataset
        self.clean_dataset = clean_dataset
        self.mixture_dataset_keys = mixture_dataset_keys
        self.apply_normalization = apply_normalization

        print(f"The limit is {limit}.")
        print(f"The offset is {offset}.")
        print(f"The len of fully dataset is {self.length}.")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        sample_length = 16384

        name = self.mixture_dataset_keys[item]
        num = name.split("_")[0]
        mixture = self.mixture_dataset[name]
        clean = self.clean_dataset[num]

        assert mixture.shape == clean.shape

        # 定长采样
        mixture, clean = sample_fixed_length_data_aligned(mixture, clean, sample_length)
        if self.apply_normalization:
            mixture = apply_mean_std(mixture)
            clean = apply_mean_std(clean)

        return mixture.reshape(1, -1), clean.reshape(1, -1), name
