import time

import librosa
import librosa.display
import numpy as np
import torch

from utils.metrics import compute_STOI, compute_PESQ
from trainer.base_trainer import BaseTrainer
from utils.utils import set_requires_grad, ExecutionTime
import matplotlib.pyplot as plt


plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            optim,
            train_dl,
            validation_dl,
            test_dl
    ):
        super(Trainer, self).__init__(config, resume, model, optim)
        self.train_data_loader = train_dl
        self.validation_data_loader = validation_dl
        self.test_data_loader = test_dl


    def _set_model_train(self):
        # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/13
        self.model.train()


    def _set_model_eval(self):
        self.model.eval()


    def _train_epoch(self):
        """定义单次训练

        定义单词训练，包含
            - [x] 设置模型运行状态
            - [x] 获取输入和目标
            - [x] 计算模型的 batch 损失
            - [x] 反向传播，更新参数
            - [x] 计算单个 epoch 的平均损失

        Returns:
            单个 epoch 的平均损失
        """
        self._set_model_train()
        loss_total = 0.0
        for i, (data, target, basename_text) in enumerate(self.train_data_loader):
            data = data.to(self.dev)
            target = target.to(self.dev)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss_total += float(loss)
            loss.backward()
            self.optimizer.step()

        # https://discuss.pytorch.org/t/about-the-relation-between-batch-size-and-length-of-data-loader/10510/4
        # The length of the loader will adapt to the batch_size
        loss_ave = loss_total / len(self.train_data_loader)
        return loss_ave


    def _valid_epoch(self):
        """单次验证

        包含：
            - 设置模型运行状态
            - 获取输入和目标
            - 计算平均损失
        """

        self._set_model_eval()
        loss_total = 0.0
        with torch.no_grad():
            for i, (data, target, basename_text) in enumerate(self.validation_data_loader):
                data = data.to(self.dev)
                target = target.to(self.dev)

                output = self.model(data)
                loss = self.loss(output, target)
                loss_total += float(loss)

        loss_ave = loss_total / len(self.validation_data_loader)

        return loss_ave


    def _test_epoch(self, epoch):
        """测试轮

        测试时使用测试集，batch_size 与 num_workers 均为 1，将每次测试后的结果保存至数组，最终返回数组，后续用于可视化

        """

        self._set_model_eval()
        # clean_ys = []
        # noisy_ys = []
        # denoisy_ys = []
        # basename_texts = []

        with torch.no_grad():
            for i, (data, target, basename_text) in enumerate(self.test_data_loader):
                data = data.to(self.dev)
                target = target.to(self.dev)
                output = self.model(data)

                self.viz.writer.add_audio(f"语音文件/{basename_text[0]}带噪语音", data, epoch, sample_rate=16000)
                self.viz.writer.add_audio(f"语音文件/{basename_text[0]}降噪语音", output, epoch, sample_rate=16000)
                self.viz.writer.add_audio(f"语音文件/{basename_text[0]}纯净语音", target, epoch, sample_rate=16000)

                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([data, output, target]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        torch.mean(y),
                        torch.std(y),
                        torch.max(y),
                        torch.min(y)
                    ))
                    librosa.display.waveplot(y.cpu().squeeze().numpy(), sr=16000, ax=ax[j])
                plt.tight_layout()

                self.viz.writer.add_figure(f"语音波形图像/{basename_text}", fig, epoch)


    def _transform_pesq_range(self, pesq_score):
        """平移 PESQ 评价指标
        将 PESQ 评价指标的范围从 -0.5 ~ 4.5 平移为 0 ~ 1

        Args:
            pesq_score: PESQ 得分

        Returns:
            0 ~ 1 范围的 PESQ 得分
        """

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
            print(f"============ Train epoch = {epoch} ============")
            print("[0 seconds] 开始训练...")
            timer = ExecutionTime()
            self.viz.set_epoch(epoch)

            train_loss = self._train_epoch()
            print("训练损失：", train_loss)
            self.viz.writer.add_scalar("训练损失", train_loss, epoch)
            valid_loss = self._valid_epoch()
            self.viz.writer.add_scalar("验证损失", valid_loss, epoch)

            # 测试一轮，并绘制波形文件
            self._test_epoch(epoch)

            if train_loss < self.mini_loss:
                self._save_checkpoint(epoch, save_best=True)

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
