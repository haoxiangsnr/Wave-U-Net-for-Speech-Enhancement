import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
from utils.utils import compute_STOI, compute_PESQ
plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for i, (mixture, clean, name) in enumerate(self.train_data_loader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            enhanced = self.model(mixture)
            loss = self.loss_function(clean, enhanced)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        self.viz.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        sample_length = self.train_data_loader.dataset.sample_length
        stoi_c_n = []
        stoi_c_d = []
        pesq_c_n = []
        pesq_c_d = []

        for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
            name = name[0]

            # [batch_size, 1, T]
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # Input is fixed length
            mixture_chunks = torch.split(mixture, sample_length, dim=2)
            if mixture_chunks[-1].shape[-1] != sample_length:
                mixture_chunks = mixture_chunks[:-1]

            enhanced_chunks = []
            for chunk in mixture_chunks:
                enhanced_chunks.append(self.model(chunk).detach().cpu())

            enhanced = torch.cat(enhanced_chunks, dim=2)

            # Back to numpy array
            mixture = mixture.cpu().numpy().reshape(-1)
            enhanced = enhanced.numpy().reshape(-1)
            clean = clean.cpu().numpy().reshape(-1)

            min_len = min(len(mixture), len(clean), len(enhanced))

            mixture = mixture[:min_len]
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]

            if i <= self.visualize_audio_limit:
                # Audio
                self.viz.writer.add_audio(f"{name}Mixture", mixture, epoch, sample_rate=16000)
                self.viz.writer.add_audio(f"{name}Enhanced", enhanced, epoch, sample_rate=16000)
                self.viz.writer.add_audio(f"{name}Clean", clean, epoch, sample_rate=16000)

            if i <= self.visualize_waveform_limit:
                # Waveform
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveplot(y, sr=16000, ax=ax[j])
                plt.tight_layout()
                self.viz.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Metric
            stoi_c_n.append(compute_STOI(clean, mixture, sr=16000))
            stoi_c_d.append(compute_STOI(clean, enhanced, sr=16000))
            pesq_c_n.append(compute_PESQ(clean, mixture, sr=16000))
            pesq_c_d.append(compute_PESQ(clean, enhanced, sr=16000))

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.viz.writer.add_scalars(f"评价指标均值/STOI", {
            "clean 与 noisy": get_metrics_ave(stoi_c_n),
            "clean 与 denoisy": get_metrics_ave(stoi_c_d)
        }, epoch)
        self.viz.writer.add_scalars(f"评价指标均值/PESQ", {
            "clean 与 noisy": get_metrics_ave(pesq_c_n),
            "clean 与 denoisy": get_metrics_ave(pesq_c_d)
        }, epoch)

        score = (get_metrics_ave(stoi_c_d) + self._transform_pesq_range(get_metrics_ave(pesq_c_d))) / 2
        return score
