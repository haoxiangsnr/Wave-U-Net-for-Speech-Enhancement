import json
import os
from pathlib import Path

import torch

from utils.visualization import TensorboardXWriter


class BaseTrainer:
    def __init__(self, config, resume: bool, model, loss_func, optim):
        """
        构建模型训练器的基类，包含以下功能：
            - 初始化 CUDA 与并行
            - 初始化 模型与优化器
            - 导入参数
            - 存储模型断点，加载模型断点
        Args:
            config: 配置文件
            resume: 本次实验是否接最近一次的断点继续运行
            model: 模型
            optim: 优化器
        """

        self.n_gpu = config["n_gpu"]
        self.dev = self._prepare_device(self.n_gpu, use_cudnn=config["use_cudnn"])

        self.model = model.to(self.dev)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.n_gpu)))

        self.optimizer = optim
        self.loss_func = loss_func
        self.epochs = config["trainer"]["epochs"]
        self.save_period = config["trainer"]["save_period"]
        self.start_epoch = 1  # 非配置项，当 resume == True 时，参数会被重置
        self.best_score = 0.0  # 非配置项
        self.root_dir = Path(config["save_location"]) / config["name"]
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.tensorboardX_logs_dir = self.root_dir / "logs"
        self._prepare_empty_dir([self.root_dir, self.checkpoints_dir, self.tensorboardX_logs_dir], resume)
        self.viz = TensorboardXWriter(self.tensorboardX_logs_dir.as_posix())

        if resume: self._resume_checkpoint()

        print("模型，优化器，参数，目录初始化完毕，本实验中使用的配置信息如下：")
        print(json.dumps(config, indent=2, sort_keys=False))
        config_save_path = os.path.join(self.root_dir, "config.json")
        with open(config_save_path, "w") as handle:
            json.dump(config, handle, indent=2, sort_keys=False)
        self._print_networks([self.model])


    def _resume_checkpoint(self):
        """恢复至最近一次的模型断点

        恢复至最近一次的模型断点。模型加载时要特殊留意，如果模型是 DataParallel 的实例，需要读取 model.module.*

        """

        latest_model_path = self.checkpoints_dir / "latest_model.tar"

        if latest_model_path.exists():
            print(f"正在加载最近一次保存的模型断点 {latest_model_path}")

            checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.dev)
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_score = checkpoint["best_score"]
            self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"])

            print("断点已被加载，将从 epoch = {} 处开始训练.".format(self.start_epoch))
        else:
            print(f"{latest_model_path} 不存在，无法加载最近一次保存的模型断点")


    def _save_checkpoint(self, epoch, is_best=False):
        """存储模型断点

        将模型断点存储至 checkpoints 目录，包含：
            - 当前轮次数
            - 历史上最高的得分
            - 优化器参数
            - 模型参数

        Args:
            epoch:
            is_best:

        """

        print("正在存储模型断点，epoch = {} ...".format(epoch))

        # 构建待存储的数据字典
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optim_state_dict": self.optimizer.state_dict(),
        }
        if self.dev.type == "cuda" and self.n_gpu > 1:
            state_dict["model_state_dict"] = self.model.module.cpu().state_dict()
        else:
            state_dict["model_state_dict"] = self.model.cpu().state_dict()

        # 存储三个数据字典
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())
        torch.save(state_dict, (self.checkpoints_dir / f"model_{str(epoch).zfill(3)}.tar").as_posix())
        if is_best:
            print("发现最优模型，正在存储中， epoch = {} ...".format(epoch))
            torch.save(state_dict, (self.checkpoints_dir / "best_model.tar").as_posix())

        self.model.to(self.dev)  # model.cpu() 会将模型转移至 CPU，此时需要将模型重新转至 GPU


    @staticmethod
    def _print_networks(nets: list):
        num_params = 0
        for net in nets:
            for param in net.parameters():
                num_params += param.numel()

        print(f"当前网络共有参数：{num_params / 1e6} 百万个.")


    @staticmethod
    def _prepare_empty_dir(dirs, resume):
        for dir_path in dirs:
            if resume:
                assert dir_path.exists()
            else:
                dir_path.mkdir(parents=False, exist_ok=True)


    @staticmethod
    def _prepare_device(n_gpu: int, use_cudnn=True):
        """
        根据 n_gpu 的大小选择使用 CPU 或 GPU

        Args:
            n_gpu(int): 实验使用 GPU 的数量，当 n_gpu 为 0 时，使用 CPU，n_gpu > 1时，使用

        Note:
            1. 运行 train.py 脚本时需要设置可见 GPU，此时修改第一块 GPU 的起始位置，否则默认加载模型只能在绝对位置的 cuda:0 上
            2. 在当前项目初始时设置可见的 GPU 后，项目中只能使用相对编号
            3. cudnn benchmark 会自动寻找算法来优化固定大小的输入时的计算，如果考虑实验的可重复性，可以设置：
                torch.backends.cudnn.deterministic = True
               即使用固定的算法，会有一些性能的影响，但是能保证可重复性
        """
        use_cpu = False

        if n_gpu == 0:
            use_cpu = True
            print("实验将使用 CPU.")
        else:
            assert n_gpu <= torch.cuda.device_count(), \
                f"使用 GPU 数量为 {n_gpu}，大于系统拥有的 GPU 数量 {torch.cuda.device_count()}"

            if use_cudnn:
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True

            print(f"实验使用的 GPU 数量为 {n_gpu}.")

        device = torch.device("cpu" if use_cpu else "cuda:0")

        return device


    def _train_epoch(self, epoch):
        raise NotImplementedError


    def train(self):
        raise NotImplementedError
