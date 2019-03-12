import os
from pathlib import Path

import librosa
from torch.utils.data import Dataset
from utils.utils import find_aligned_wav_files, sample_fixed_length_data_aligned


class TrainDataset(Dataset):
    """
    定义训练集
    """

    def __init__(self, dataset_dir, limit=10, offset=0):
        """
        构建训练数据集
        Args:
            dataset_dir (str): 验证数据集根目录，必须包含 noisy 和 clean 子目录
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
        """
        noisy_dir = Path(dataset_dir) / "noisy"
        clean_dir = Path(dataset_dir) / "clean"

        assert noisy_dir.exists(), "数据目录下必须包含 noisy 子目录"
        assert clean_dir.exists(), "数据目录下必须包含 clean 子目录"

        self.noisy_wav_paths, self.clean_wav_paths, self.length = find_aligned_wav_files(
            noisy_dir.as_posix(), clean_dir.as_posix(), limit=limit, offset=offset
        )

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        sr = 16000

        noisy_wav_path = self.noisy_wav_paths[item]
        clean_wav_path = self.clean_wav_paths[item]

        noisy_y, _ = librosa.load(noisy_wav_path, sr=sr)
        clean_y, _ = librosa.load(clean_wav_path, sr=sr)

        basename_text, _ = os.path.splitext(os.path.basename(noisy_wav_path))

        # 定长采样
        noisy_y, clean_y = sample_fixed_length_data_aligned(noisy_y, clean_y, sr)

        return noisy_y, clean_y, basename_text