import numpy as np
import os
from pathlib import Path

import librosa
from torch.utils.data import Dataset
from utils.utils import find_aligned_wav_files, sample_fixed_length_data_aligned

class TrainNpyDataset(Dataset):
    """
    定义训练集
    """

    def __init__(self, dataset, limit=10, offset=0, for_train=True):
        """
        构建训练数据集
        Args:
            dataset (str): 验证数据集 NPY 文件
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
        """
        mark = "train.npy" if for_train else "test.npy"
        dataset = os.path.join(dataset, mark)
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
        np.random.seed(0)
        sample_length = 16384

        key = self.keys[item]
        value = self.dataset_dict[key]

        noisy_y = value["noisy"]
        clean_y = value["clean"]

        assert noisy_y.shape == clean_y.shape

        # 定长采样
        noisy_y, clean_y = sample_fixed_length_data_aligned(noisy_y, clean_y, sample_length)

        return noisy_y.reshape(1, -1), clean_y.reshape(1, -1), key


class TrainDataset(Dataset):
    """
    定义训练集
    """

    def __init__(self, dataset, limit=10, offset=0, for_train=True):
        """
        构建训练数据集
        Args:
            dataset (str): 验证数据集根目录，必须包含 noisy 和 clean 子目录
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
        """
        mark = "train" if for_train else "test"
        noisy_dir = Path(dataset) / mark / "noisy"
        clean_dir = Path(dataset) / mark /"clean"

        assert noisy_dir.exists(), "数据目录下必须包含 noisy 子目录"
        assert clean_dir.exists(), "数据目录下必须包含 clean 子目录"

        self.noisy_wav_paths, self.clean_wav_paths, self.length = find_aligned_wav_files(
            noisy_dir.as_posix(), clean_dir.as_posix(), limit=limit, offset=offset
        )


    def set_catch(self):
        """
        读取语音信号，并缓存为文件，
        Returns:

        """
        pass


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        np.random.seed(0)
        sr = 16000
        sample_length = 16384

        noisy_wav_path = self.noisy_wav_paths[item]
        clean_wav_path = self.clean_wav_paths[item]

        noisy_y, _ = librosa.load(noisy_wav_path, sr=sr)
        clean_y, _ = librosa.load(clean_wav_path, sr=sr)

        basename_text, _ = os.path.splitext(os.path.basename(noisy_wav_path))

        # 定长采样
        noisy_y, clean_y = sample_fixed_length_data_aligned(noisy_y, clean_y, sample_length)

        return noisy_y.reshape(1, -1), clean_y.reshape(1, -1), basename_text