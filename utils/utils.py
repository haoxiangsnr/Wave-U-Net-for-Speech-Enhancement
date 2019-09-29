import importlib
import time
import os

import torch
from pypesq import pesq
import numpy as np
from pystoi.stoi import stoi


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg):
    """
    According to config items, load specific module dynamically with params.

    eg，config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }

    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])
    return getattr(module, module_cfg["main"])(**module_cfg["args"])


def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    """
    使用 pypesq 计算 pesq 评价指标。
    Notes：
        pypesq 是 PESQ 官方代码（C 语言）的 wrapper。官方代码在某些语音上会报错，而其报错时直接调用了 exit() 函数，直接导致 Python 脚本停止运行，亦无法捕获异常，实在过于粗鲁。
        为了防止 Python 脚本被打断，这里使用子进程执行 PESQ 评价指标的计算，设置子进程的超时。
        设置子进程超时的方法：https://stackoverflow.com/a/29378611
    Returns:
        当语音可以计算 pesq score 时，返回 pesq score，否则返回 -1
    """
    return pesq(clean_signal, noisy_signal, sr)
    # pool = multiprocessing.Pool(1)`
    # rslt = pool.apply_async(_compute_PESQ_sub_task, args = (clean_signal, noisy_signal, sr))
    # pool.close() # 关闭进程池，不运行再向内添加进程
    # rslt.wait(timeout=1) # 子进程的处理时间上限为 1 秒钟，到时未返回结果，则终止子进程
    #
    # if rslt.ready(): # 在 1 秒钟内子进程处理完毕
    #     return rslt.get()
    # else: # 过了 1 秒了，但仍未处理完，返回 -1
    #     return -1


def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min


def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """
    sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)

    start = np.random.randint(frames_total - sample_length + 1)
    # print(f"Random crop from: {start}")
    end = start + sample_length

    return data_a[start:end], data_b[start:end]


def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)


def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\tmax: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")
