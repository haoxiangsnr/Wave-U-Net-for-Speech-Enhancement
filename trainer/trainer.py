import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
from utils.metrics import compute_STOI, compute_PESQ
from utils.utils import ExecutionTime

plt.switch_backend('agg')

class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optim,
            train_dl,
            validation_dl,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optim)
        self.train_data_loader = train_dl
        self.validation_data_loader = validation_dl


    def _set_model_train(self):
        # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/13
        self.model.train()


    def _set_model_eval(self):
        self.model.eval()


    def _train_epoch(self, epoch):
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
            loss = self.loss_function(output, target)
            loss_total += float(loss)
            loss.backward()
            self.optimizer.step()

        # https://discuss.pytorch.org/t/about-the-relation-between-batch-size-and-length-of-data-loader/10510/4
        # The length of the loader will adapt to the batch_size
        dl_len = len(self.train_data_loader)
        visualize_loss = lambda tag, total: self.viz.writer.add_scalar(f"训练损失/{tag}", total / dl_len, epoch)
        visualize_loss("loss", loss_total)


    def _test_epoch(self, epoch):
        """测试轮

        测试时使用测试集，batch_size 与 num_workers 均为 1，将每次测试后的结果保存至数组，最终返回数组，后续用于可视化

        """

        self._set_model_eval()
        stoi_c_n = []
        stoi_c_d = []
        pesq_c_n = []
        pesq_c_d = []

        with torch.no_grad():
            for i, (data, target, basename_text) in enumerate(self.validation_data_loader):
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

                ny = data.cpu().numpy().reshape(-1)
                dy = output.cpu().numpy().reshape(-1)
                cy = target.cpu().numpy().reshape(-1)

                stoi_c_n.append(compute_STOI(cy, ny, sr=16000))
                stoi_c_d.append(compute_STOI(cy, dy, sr=16000))
                pesq_c_n.append(compute_PESQ(cy, ny, sr=16000))
                pesq_c_d.append(compute_PESQ(cy, dy, sr=16000))

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

    def _transform_pesq_range(self, pesq_score):
        """平移 PESQ 评价指标
        将 PESQ 评价指标的范围从 -0.5 ~ 4.5 平移为 0 ~ 1

        Args:
            pesq_score: PESQ 得分

        Returns:
            0 ~ 1 范围的 PESQ 得分
        """

        return (pesq_score + 0.5) * 2 / 10


    def _is_best_score(self, score):
        """检查当前的结果是否为最佳模型"""
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

            self._train_epoch(epoch)

            if self.validation_period!= 0 and epoch % self.validation_period == 0:
                # 测试一轮，并绘制波形文件
                print(f"[{timer.duration()} seconds] 训练结束，开始计算评价指标...")
                score = self._test_epoch(epoch)

                if self._is_best_score(score):
                    self._save_checkpoint(epoch, is_best=True)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

            print(f"[{timer.duration()} seconds] 完成当前 Epoch.")
