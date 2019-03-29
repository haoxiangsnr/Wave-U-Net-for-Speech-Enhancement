import argparse
import json
from pathlib import Path

import librosa
import tablib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.test_npy_dataset import TestNpyDataset
from models.unet import UNet
from utils.metrics import compute_STOI, compute_PESQ


def load_checkpoint(checkpoints_dir, name, dev):
    checkpoint_path = checkpoints_dir / name
    assert checkpoint_path.exists(), f"模型断点 {name} 不存在"
    checkpoint = torch.load(checkpoint_path.as_posix(), map_location=dev)
    return checkpoint


def main(config, epoch):
    test_data_args = config["test_data"]
    test_dataset = TestNpyDataset(
        dataset=test_data_args["dataset"],
        limit=test_data_args["limit"],
        offset=test_data_args["offset"]
    )
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=1,
    )

    dev = torch.device("cpu")
    model = UNet()

    root_dir = Path(config["save_location"]) / config["name"]
    checkpoints_dir = root_dir / "checkpoints"

    if isinstance(epoch, str):
        if epoch == "latest":
            checkpoint = load_checkpoint(checkpoints_dir, "latest_model.tar", dev)
            model_state_dict = checkpoint["model_state_dict"]
            print(f"Load latest checkpoints, is {checkpoint['epoch']}")
        elif epoch == "best":
            checkpoint = load_checkpoint(checkpoints_dir, "best_model.tar", dev)
            model_state_dict = checkpoint["model_state_dict"]
            print(f"Load best checkpoints, is {checkpoint['epoch']}")
        else:
            raise ValueError(f"指定的 epoch 参数存在问题，epoch 为 {epoch}.")
    else:
        checkpoint = load_checkpoint(checkpoints_dir, "latest_model.tar", dev)
        model_state_dict = checkpoint["model_state_dict"]
        print(f"Load latest checkpoints, is {epoch}")

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    model.to(dev)
    model.load_state_dict(model_state_dict)
    model.eval()

    print("加载模型成功，开始计算评价指标.")

    results_dir = root_dir / f"{config['name']}_{epoch}_results"
    results_dir.mkdir(parents=False, exist_ok=True)

    headers = ("语音编号", "噪声类型", "信噪比",
               "STOI 纯净与带噪", "STOI 纯净与降噪 ",
               "PESQ 纯净与带噪", "PESQ 纯净与降噪",
               "STOI 提升",
               "PESQ 提升")  # 定义导出为 Excel 文件的格式
    metrics_seq = []

    with torch.no_grad():
        for i, (ny, cy, basename_text) in tqdm(enumerate(test_data_loader), desc="正在计算评价指标中："):
            basename_text = basename_text[0]
            dy = model(ny)

            dy = dy.numpy().reshape(-1)
            ny = ny.numpy().reshape(-1)
            cy = cy.numpy().reshape(-1)

            stoi_c_n = compute_STOI(cy, ny, sr=16000)
            stoi_c_d = compute_STOI(cy, dy, sr=16000)
            pesq_c_n = compute_PESQ(cy, ny, sr=16000)
            pesq_c_d = compute_PESQ(cy, dy, sr=16000)

            num, noise, snr = basename_text.split("_")
            metrics_seq.append((
                num, noise, snr,
                stoi_c_n, stoi_c_d,
                pesq_c_n, pesq_c_d,
                (stoi_c_d - stoi_c_n) / stoi_c_n,
                (pesq_c_d - pesq_c_n) / pesq_c_n
            ))

            for type in ["clean", "denoisy", "noisy"]:
                output_path = results_dir / f"{basename_text}_{type}.wav"
                if type == "clean":
                    librosa.output.write_wav(output_path.as_posix(), cy, 16000)
                elif type == "denoisy":
                    librosa.output.write_wav(output_path.as_posix(), dy, 16000)
                else:
                    librosa.output.write_wav(output_path.as_posix(), ny, 16000)

    data = tablib.Dataset(*metrics_seq, headers=headers)
    metrics_save_dir = root_dir / f"{config['name']}_{epoch}.xls"
    print(f"测试过程结束，正在将结果存储至 {metrics_save_dir.as_posix()}")
    with open(metrics_save_dir.as_posix(), 'wb') as f:
        f.write(data.export('xls'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="[TEST] Speech Ehancement")
    parser.add_argument("-C", "--config", required=True, type=str)
    parser.add_argument("-E", "--epoch", default="best", help="'best' | 'latest' | int(epoch)")
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config, args.epoch)
