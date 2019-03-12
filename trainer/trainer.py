import time

import numpy as np
import torch

from utils.metrics import compute_STOI, compute_PESQ
from trainer.base_trainer import BaseTrainer
from utils.utils import set_requires_grad, ExecutionTime


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            optim,
            train_dl,
            validation_dl=None
    ):
        super(Trainer, self).__init__(config, resume, model, optim)
        self.train_data_loader = train_dl
        self.validation_data_loader = validation_dl

    def _set_model_train(self):
        # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/13
        self.model.train()

    def _set_model_eval(self):
        self.model.eval()

    def _train_epoch(self):
        """定义单次训练的逻辑"""
        self._set_model_train()
        losses = {
            "D_fake": 0.0,
            "D_real": 0.0,
            "D": 0.0,
            "G_GAN": 0.0,
            "G_L1": 0.0,
            "G": 0.0,
        }
        # TODO
        return losses

    def _valid_epoch(self, valid_data_loader):
        """定义单次验证的逻辑"""
        # TODO
        self._set_model_eval()
        with torch.no_grad():
            pass

    def _transform_pesq_range(self, pesq_score):
        """将 pesq 的范围从 -0.5 ~ 4.5 迁移至 0 ~ 1"""
        return (pesq_score + 0.5) * 2 / 10

    def _visualization_epoch(self):
        """定义需要可视化的项"""
        self.viz.visualize_factors()
        self.viz.visualize_metrics()
        self.viz.visualize_wav_files()
        self.viz.visualize_wav_waveform()
        self.viz.visualize_mel_spectrograms()

    def _is_best_score(self, metrics):
        """检查当前的结果是否为最佳模型"""
        stoi_score = np.mean(np.array(metrics["stoi"]["clean_and_denoisy_values"]))
        pesq_score = np.mean(np.array(metrics["pesq"]["clean_and_denoisy_values"]))

        score = (stoi_score + self._transform_pesq_range(pesq_score)) / 2

        if score >= self.best_score:
            self.best_score = score
            return True
        return False

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            timer = ExecutionTime()
            print(f"============ Train epoch = {epoch} ============")
            self.viz.set_epoch(epoch)

            print("[0 seconds] 开始训练...")
            losses = self._train_epoch()
            # TODO
            # 存储常规模型
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

            print(f"[{timer.duration()} seconds] 完成当前 Epoch.")

    @staticmethod
    def _eval_metrics(clean_signals, denoisy_signals, noisy_signals, sr=16000):
        """
        stoi: 0 ~ 1
        pesq: -0.5 ~ 4.5，从 -0.5 ~ 4.5 的取值范围

        Args:
            noisy (1D ndarray): 纯净语音的信号
            denoisy (1D ndarray): 降噪语音的信号（或带噪）
            sr (1D ndarray): 采样率，默认值为 16000

        Returns:
            返回评价结果将直接送到可视化工具中可视化
        """
        print("正在计算评价指标（eavl_metrics）... ")
        stoi_clean_and_noisy_values = []
        stoi_clean_and_denoisy_values = []
        pesq_clean_and_noisy_values = []
        pesq_clean_and_denoisy_values = []

        for clean, denoisy, noisy in zip(clean_signals, denoisy_signals, noisy_signals):
            stoi_clean_and_noisy_values.append(compute_STOI(clean, noisy, sr=sr))
            stoi_clean_and_denoisy_values.append(compute_STOI(clean, denoisy, sr=sr))

            pesq_clean_and_noisy_values.append(compute_PESQ(clean, noisy, sr=sr))
            pesq_clean_and_denoisy_values.append(compute_PESQ(clean, denoisy, sr=sr))

        return {
            "stoi": {
                "clean_and_noisy_values": stoi_clean_and_noisy_values,
                "clean_and_denoisy_values": stoi_clean_and_denoisy_values,
            },
            "pesq": {
                "clean_and_noisy_values": pesq_clean_and_noisy_values,
                "clean_and_denoisy_values": pesq_clean_and_denoisy_values,
            },
        }