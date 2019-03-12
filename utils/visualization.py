import json

import librosa
import matplotlib.pyplot as plt
import tensorboardX
import numpy as np
import librosa.display

plt.switch_backend('agg')

class TensorboardXWriter:
    def __init__(self, tensorboardX_logs_dir):
        self.writer = tensorboardX.SummaryWriter(tensorboardX_logs_dir)