# UNet-Time-Domain

End to end speech enhancement 

## Train

```json
{
    "name": "chime_v2",
    "n_gpu": 1,
    "use_cudnn": true,
    "loss_func": "mse_loss",
    "model_arch": "unet_with_diff_dilation",
    "save_location": "/media/imucs/DataDisk/haoxiang/Experiment/UNet",
    "dataset": "/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/chime/",
    "description": "上次训练 **chime_v2** 数据失败了，本次继续训练，使用 xshell，并使用 xshell 测试",
    "visualize_metrics": false,
    "use_npy": true,
    "train_data": {
        "limit": 0,
        "offset": 0,
        "batch_size": 150,
        "shuffle": true,
        "num_workers": 40
    },
    "valid_data": {
        "limit": 400,
        "offset": 9000,
        "batch_size": 400,
        "num_workers": 40
    },
    "test_data": {
        "limit": 200,
        "offset": 0
    },
    "optimizer": {
        "lr": 0.0002
    },
    "trainer": {
        "epochs": 600,
        "save_period": 3
    },
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    }
}
```

## Test

- 同时运行两个测试完全可行
    - Wav file saved as `{save_dir} / {name}_{epoch}_{results} / {basename}_{text}_{type}.wav`
    - Excel file saved as `{name}_{epoch}.xls`

```shell
# 指定配置文件
python test.py -C config/test_diff_dilation.json
```

```json
{
    "name": "unet_with_diff_dilation",
    "model_arch": "unet_with_diff_dilation",
    "save_location": "/media/imucs/DataDisk/haoxiang/Experiment/UNet",
    "dataset": "/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/release_0_-5_-10_-15_800_900",
    "test_data": {
        "limit": 10000,
        "offset": 44000
    }
}
```