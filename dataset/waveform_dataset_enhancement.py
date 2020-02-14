import os
from torch.utils.data import Dataset
import librosa
from utils.utils import sample_fixed_length_data_aligned


class WaveformDataset(Dataset):
    def __init__(self, dataset, limit=None, offset=0, sample_length=16384, train=True):
        """
        构建增强数据集
        Args:
            dataset (str): 语音数据集的路径，拓展名为 txt，见 Notes 部分
            limit (int): 数据集的数量上限
            offset (int): 数据集的起始位置的偏移值
            sample_length(int): 模型仅支持定长输入，这个参数指定了每次输入模型的大小

        Notes:
            语音数据集格式如下：
            <带噪语音1的绝对路径>
            <带噪语音2的绝对路径>
            ...
            <带噪语音n的绝对路径>

            eg:
            /enhancement/noisy/a.wav
            /enhancement/noisy/b.wav
            ...

        Return:
            (mixture signals, clean signals, file name)
        """
        super(WaveformDataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]
        

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        mixture_path = self.dataset_list[item]
        name = os.path.splitext(os.path.basename(mixture_path))[0]

        mixture, _ = librosa.load(os.path.abspath(os.path.expanduser(mixture_path)), sr=None)

        return mixture.reshape(1, -1), name
