import argparse
import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

import data.train_dataset
import data.test_dataset
import models as model_arch
import models.loss as model_loss
from trainer.trainer import Trainer

torch.manual_seed(0)
np.random.seed(0)

def main(config, resume):
    if config["use_npy"]:
        TrainDataset = data.train_dataset.TrainNpyDataset
        TestDataset = data.test_dataset.TestNpyDataset
    else:
        TrainDataset = data.train_dataset.TrainDataset
        TestDataset = data.test_dataset.TestDataset

    train_data_args = config["train_data"]
    train_dataset = TrainDataset(
        dataset=config["dataset"],
        limit=train_data_args["limit"],
        offset=train_data_args["offset"]
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_data_args["batch_size"],
        num_workers=train_data_args["num_workers"],
        shuffle=train_data_args["shuffle"],
        pin_memory=True
    )

    valid_data_args = config["valid_data"]
    valid_dataset = TrainDataset(
        dataset=config["dataset"],
        limit=valid_data_args["limit"],
        offset=valid_data_args["offset"]
    )
    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_data_args["batch_size"],
        num_workers=valid_data_args["num_workers"],
        pin_memory=True
    )

    test_data_args = config["test_data"]
    test_dataset = TestDataset(
        dataset=config["dataset"],
        limit=test_data_args["limit"],
        offset=test_data_args["offset"],
    )
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=1
    )

    model = getattr(model_arch, config["model_arch"]).UNet()

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(0.9, 0.999)
    )

    loss_func = getattr(model_loss, config["loss_func"])

    trainer = Trainer(
        config=config,
        resume=resume,
        model=model,
        loss_func=loss_func,
        optim=optimizer,
        train_dl=train_data_loader,
        validation_dl=valid_data_loader,
        test_dl=test_data_loader
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

    main(config, resume=args.resume)
