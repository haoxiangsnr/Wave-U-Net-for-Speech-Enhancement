import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from data.train_npy_dataset import TrainNpyDataset
from models.unet import UNet
from trainer.trainer import Trainer


def main(config, resume):
    train_data_args = config["train_data"]
    train_dataset = TrainNpyDataset(
        dataset=train_data_args["dataset"],
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
    valid_dataset = TrainNpyDataset(
        dataset=valid_data_args["dataset"],
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
    test_dataset = TrainNpyDataset(
        dataset=test_data_args["dataset"],
        limit=test_data_args["limit"],
        offset=test_data_args["offset"],
    )
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=1
    )

    model = UNet()

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(0.9, 0.999)
    )

    trainer = Trainer(
        config=config,
        resume=resume,
        model=model,
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
