import matplotlib.pyplot as plt
import tensorboardX


plt.switch_backend('agg')


class TensorboardXWriter:
    def __init__(self, tensorboardX_logs_dir):
        self.writer = tensorboardX.SummaryWriter(tensorboardX_logs_dir)
        self.epoch = 0


    def set_epoch(self, epoch):
        self.epoch = epoch
