# UNet-Time-Domain

End to end speech enhancement 

## Train

```json
{
    "name": "unet_basic",
    "n_gpu": 1,
    "use_cudnn": true,
    "loss_func": "mse_loss",
    "model_arch": "unet",
    "save_location": "/media/imucs/DataDisk/haoxiang/Experiment/UNet",
    "dataset": "/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/release_0_-5_-10_-15_800_900",
    "description": "",
    "visualize_metrics_period": 10,
    "use_npy": true,
    "train_data": {
        "limit": 0,
        "offset": 0,
        "batch_size": 170,
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
        "epochs": 1000,
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

```shell
# Specify test json file
python test.py -C config/test_diff_dilation.json
```

```json
{
    "name": "unet_basic",
    "model_arch": "unet",
    "save_location": "/media/imucs/DataDisk/haoxiang/Experiment/UNet",
    "dataset": "/media/imucs/DataDisk/haoxiang/Release/speech_enhancement/release_0_-5_-10_-15_800_900",
    "test_data": {
        "limit": 0,
        "offset": 44000
    }
}
```

- Wav file saved as `{save_dir} / {name}_{epoch}_{results} / {type} / {basename_text}.wav`
- Excel file saved as `{name}_{epoch}.xls`


## ToDo

- [x] Separate computing metrics from test script