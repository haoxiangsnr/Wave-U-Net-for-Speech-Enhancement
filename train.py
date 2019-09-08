import argparse
import os

parser = argparse.ArgumentParser(description='Self-attention for Speech Enhancement')
parser.add_argument("-C", "--config", required=True, type=str,
                    help="Specify the configuration file for training (*.json).")
parser.add_argument('-D', '--device', default=None, type=str,
                    help="Specify the GPUs visible in the experiment, e.g. '1,2,3'.")
parser.add_argument("-R", "--resume", action="store_true",
                    help="Whether to resume training from a recent breakpoint.")
args = parser.parse_args()
if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import numpy as np
from torch.utils.data import DataLoader
import torch
import json

from trainer.trainer import Trainer
from utils.utils import initialize_config

config = json.load(open(args.config))
config["experiment_name"] = os.path.splitext(os.path.basename(args.config))[0]
config["train_config_path"] = args.config


def main(config, resume):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    train_dataset = initialize_config(config["train_dataset"])
    train_data_loader = DataLoader(
        dataset=train_dataset,
        shuffle=config["train_dataloader"]["shuffle"],
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"]
    )

    validation_dataset = initialize_config(config["validation_dataset"])
    valid_data_loader = DataLoader(
        dataset=validation_dataset,
        num_workers=config["validation_dataloader"]["num_workers"],
        batch_size=config["validation_dataloader"]["batch_size"]
    )

    model = initialize_config(config["model"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], 0.999)
    )

    loss_function = initialize_config(config["loss_function"])

    trainer = Trainer(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_data_loader,
        validation_dataloader=valid_data_loader
    )

    trainer.train()


main(config, resume=args.resume)
