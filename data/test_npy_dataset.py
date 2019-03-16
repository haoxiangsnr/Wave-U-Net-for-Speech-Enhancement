import numpy as np
import os
from pathlib import Path

import librosa
from torch.utils.data import Dataset
from utils.utils import find_aligned_wav_files, sample_fixed_length_data_aligned


class TestNpyDataset(Dataset):
    """
    定义测试集
    """


    def __init__(self, dataset, limit=10, offset=0):
        """
        构建测试数据集
        Args:
            dataset (str): 验证数据集 NPY 文件
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
        """
        assert Path(dataset).exists(), f"数据集 {dataset} 不存在"

        print(f"Loading NPY dataset {dataset} ...")
        self.dataset_dict:dict = np.load(dataset).item()

        print(f"The len of full dataset is {len(self.dataset_dict)}.")
        print(f"The limit is {limit}.")
        print(f"The offset is {offset}.")

        if limit == 0:
            limit = len(self.dataset_dict)

        self.keys = list(self.dataset_dict.keys())
        self.keys.sort()
        self.keys = self.keys[offset: offset + limit]

        self.length = len(self.keys)
        print(f"Finish, len(finial dataset) == {self.length}.")

    def __len__(self):
        return self.length


    def __getitem__(self, item):
        key = self.keys[item]
        value = self.dataset_dict[key]

        noisy_y = value["noisy"]
        clean_y = value["clean"]

        assert noisy_y.shape == clean_y.shape

        return noisy_y.reshape(1, -1), clean_y.reshape(1, -1), key
