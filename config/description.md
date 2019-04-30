# 参数描述

## yaml 与 json 对比

使用 `yaml` 格式来说明配置参数的作用，但最终使用 `json` 格式来存放

- `yaml` 可以添加注释，但通过缩进控制信息层次，比较乱，尤其是其中包含极长的路径时
- `json` 对格式有严格的控制，比如它甚至会控制最后的逗号，但它无法添加注释

## save_location 目录

- `checkpoints`：包含训练过程中产生的所有模型断点
- `logs`: 存放 TensorboardX 可视化所需的数据
- `config.json`： 本次实验的配置信息

## 训练参数汇总

```yaml
name: "UNet_first_exp"          # [str] 实验名
n_gpu: 1                        # [int] GPU数量，需要配合 train.py 脚本的 -D (--device) 参数
use_cudnn: true                 # [bool] 是否使用 Cudnn 加速训练，使用 Cudnn 可能会导致实验无法重复
save_location:                  # [str] 实验训练过程中产生的数据的存放位置，由于模型断点非常占用空间，可以将 save_location 指定在其他磁盘上。
                                # 不一定要在当前目录下。
description:                    # [str] 实验描述信息
visualize_metrics_period:       # [str] 可视化评价指标结果的间隔（包含波形文件，语音的可视化）
loss_function:            
  module: "models.loss"         # [str] 存放损失函数的模块文件
  main: "mse_loss"              # [str] 模块中可能存有多个函数或者类，main 指定需要加载的函数
  args: {}                      # 传入给 mse_loss 的位置参数
  
model:  
  module: "models.unet_basic"   # [str] 存放模型类的模块文件
  main: "UNet"                  # [str] 模块中需要加载的模型类
  args:                         # [dict] 传入给 UNet 类的位置参数
    n_layers: 11
    channels_interval: 24       
optimizer:
  lr: 0.001
trainer:
  epochs: 1000
  save_period:                  # [str] 存储模型断点的周期
train_dataset:
  mixture: "/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/noise_7_clean_900/train/mixture.npy"
  clean: "/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/noise_7_clean_900/train/clean.npy"
  limit: null
  offset: 0
  shuffle: true
  num_workers: 40
  batch_size: 150
valid_dataset:
  mixture: "/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/noise_7_clean_900/test/mixture.npy"
  clean: "/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/noise_7_clean_900/test/clean.npy"
  limit: 100
  offset: 0
```

## 测试参数汇总

TODO