import os
from torch.utils import data
import librosa
from util.utils import sample_fixed_length_data_aligned


class Dataset(data.Dataset):
    def __init__(self, dataset, limit=None, offset=0, sample_length=16384, mode="train"):
        """
        构建训练数据集
        Args:
            dataset (str): 语音数据集的路径，拓展名为 txt，见 Notes 部分
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
            sample_length(int): 模型仅支持定长输入，这个参数指定了每次输入模型的大小
            mode(str): 当为 train 时，表示需要对语音进行定长切分，当为 validation 时，表示不需要，直接返回全长的语音。

        Notes:
            语音数据集格式如下：
            <带噪语音1的路径><空格><纯净语音1的路径>
            <带噪语音2的路径><空格><纯净语音2的路径>
            ...
            <带噪语音n的路径><空格><纯净语音n的路径>

            eg:
            /train/noisy/a.wav /train/clean/a.wav
            /train/noisy/b.wav /train/clean/b.wav
            ...

        Return:
            (mixture signals, clean signals, file name)
        """
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        assert mode in ("train", "validation"), "Mode must be one of train or validation."

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mixture_path, clean_path = self.dataset_list[item].split(" ")
        name = os.path.splitext(os.path.basename(mixture_path))[0]
        mixture, _ = librosa.load(os.path.abspath(os.path.expanduser(mixture_path)), sr=None)
        clean, _ = librosa.load(os.path.abspath(os.path.expanduser(clean_path)), sr=None)

        if self.mode == "train":
            # The input of model should be fixed length.
            mixture, clean = sample_fixed_length_data_aligned(mixture, clean, self.sample_length)
            return mixture.reshape(1, -1), clean.reshape(1, -1), name
        else:
            return mixture.reshape(1, -1), clean.reshape(1, -1), name
