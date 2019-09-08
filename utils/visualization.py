from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    def __init__(self, logs_dir):
        self.writer = SummaryWriter(logs_dir, flush_secs=30)