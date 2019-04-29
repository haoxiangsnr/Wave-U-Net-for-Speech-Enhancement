import argparse
import importlib
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import models.loss as model_loss
from data.train_dataset import TrainDataset
from trainer.trainer import Trainer

torch.manual_seed(0)
np.random.seed(0)

def main(config, resume):
    train_dataset = TrainDataset(
        mixture_dataset=config["train_dataset"]["mixture"],
        clean_dataset=config["train_dataset"]["clean"],
        limit=config["train_dataset"]["limit"],
        offset=config["train_dataset"]["offset"],
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train_dataset"]["batch_size"],
        num_workers=config["train_dataset"]["num_workers"],
        shuffle=config["train_dataset"]["shuffle"]
    )

    valid_dataset = TrainDataset(
        mixture_dataset=config["valid_dataset"]["mixture"],
        clean_dataset=config["valid_dataset"]["clean"],
        limit=config["valid_dataset"]["limit"],
        offset=config["valid_dataset"]["offset"]
    )
    valid_data_loader = DataLoader(
        dataset=valid_dataset
    )

    model_cfg = config["model"]
    model_path = f"models.{model_cfg['type']}"
    model = importlib.import_module(model_path).UNet(**model_cfg["args"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(0.9, 0.999)
    )

    loss_function = getattr(model_loss, config["loss_function"])

    trainer = Trainer(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optim=optimizer,
        train_dl=train_data_loader,
        validation_dl=valid_data_loader,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNet For Speech Enhancement')
    parser.add_argument("-C", "--config", required=True, type=str, help="训练配置文件")
    parser.add_argument('-D', '--device', default=None, type=str, help="indices of GPUs to enable，e.g. '1,2,3'")
    parser.add_argument("-R", "--resume", action="store_true", help="是否从最近的一个断点处继续训练")
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # load config file
    config = json.load(open(args.config))
    config["train_config_path"] = args.config

    main(config, resume=args.resume)
