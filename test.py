import argparse
import json
from pathlib import Path

import librosa
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import models as model_arch
from data.test_dataset import TestNpyDataset

def pad_last(data, size):
    need = size - data.shape[2]
    return torch.cat((data, torch.zeros((1, 1, need))), dim=2)

def load_checkpoint(checkpoints_dir, name, dev):
    checkpoint_path = checkpoints_dir / name
    assert checkpoint_path.exists(), f"模型断点 {name} 不存在"
    checkpoint = torch.load(checkpoint_path.as_posix(), map_location=dev)
    return checkpoint


def main(config, epoch):
    test_data_args = config["test_data"]
    test_dataset = TestNpyDataset(
        dataset=config["dataset"],
        limit=test_data_args["limit"],
        offset=test_data_args["offset"]
    )
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=1
    )

    dev = torch.device("cpu")
    model = getattr(model_arch, config["model_arch"]).UNet()

    root_dir = Path(config["save_location"]) / config["name"]
    checkpoints_dir = root_dir / "checkpoints"

    if isinstance(epoch, str):
        if epoch == "latest":
            checkpoint = load_checkpoint(checkpoints_dir, "latest_model.tar", dev)
            model_state_dict = checkpoint["model_state_dict"]
            print(f"Load latest checkpoints, is {checkpoint['epoch']}.")
        elif epoch == "best":
            checkpoint = load_checkpoint(checkpoints_dir, "best_model.tar", dev)
            model_state_dict = checkpoint["model_state_dict"]
            print(f"Load best checkpoints, is {checkpoint['epoch']}.")
        else:
            checkpoint = load_checkpoint(checkpoints_dir, f"model_{str(epoch).zfill(3)}.tar", dev)
            model_state_dict = checkpoint["model_state_dict"]
            print(f"Load checkpoints is {epoch}.")
    else:
        raise ValueError(f"指定的 epoch 参数存在问题，epoch 为 {epoch}.")

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    model.to(dev)
    model.load_state_dict(model_state_dict)
    model.eval()

    results_dir = root_dir / f"{config['name']}_{epoch}_results"
    results_dir.mkdir(parents=False, exist_ok=True)

    with torch.no_grad():
        for i, (ny, cy, basename_text) in tqdm(enumerate(test_data_loader), desc="增强中"):
            basename_text = basename_text[0]
            ny_list = list(torch.split(ny, 16384, dim=2))
            last_one = pad_last(ny_list[-1], 16384)
            ny_list[-1] = last_one

            dy_list = [model(y) for y in ny_list]
            dy = torch.cat(tuple(dy_list), 2)

            dy = dy.numpy().reshape(-1)
            ny = ny.numpy().reshape(-1)
            cy = cy.numpy().reshape(-1)

            for type in ["clean", "denoisy", "noisy"]:
                (results_dir / type).mkdir(exist_ok=True)
                output_path = results_dir / type / f"{basename_text}.wav"
                if type == "clean":
                    librosa.output.write_wav(output_path.as_posix(), cy, 16000)
                elif type == "denoisy":
                    librosa.output.write_wav(output_path.as_posix(), dy[:len(cy)], 16000)
                else:
                    librosa.output.write_wav(output_path.as_posix(), ny, 16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="[TEST] Speech Enhancement")
    parser.add_argument("-C", "--config", required=True, type=str)
    parser.add_argument("-E", "--epoch", default="best", help="'best' | 'latest' | {epoch}")
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config, args.epoch)
