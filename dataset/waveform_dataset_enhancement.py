import os
from torch.utils.data import Dataset
import librosa


class WaveformDataset(Dataset):
    def __init__(self, dataset, limit=None, offset=0, sample_length=16384):
        """Construct dataset for enhancement.
        Args:
            dataset (str): *.txt. The path of the dataset list file. See "Notes."
            limit (int): Return at most limit files in the list. If None, all files are returned.
            offset (int): Return files starting at an offset within the list. Use negative values to offset from the end of the list.

        Notes:
            dataset list file：
            <noisy_1_path>
            <noisy_2_path>
            ...
            <noisy_n_path>

            e.g.
            /enhancement/noisy/a.wav
            /enhancement/noisy/b.wav
            ...

        Return:
            (mixture signals, filename)
        """
        super(WaveformDataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mixture_path = self.dataset_list[item]
        name = os.path.splitext(os.path.basename(mixture_path))[0]

        mixture, _ = librosa.load(os.path.abspath(os.path.expanduser(mixture_path)), sr=None)

        return mixture.reshape(1, -1), name
