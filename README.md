# Wave-U-Net-for-Speech-Enhancement

Implement [Wave-U-Net](https://arxiv.org/abs/1806.03185) by PyTorch, and migrate it to the speech enhancement.

![](./doc/tensorboard.png)
![](./doc/audio.png)

## Dependencies


```shell
# Make sure the /bin directory of CUDA be added to PATH enveriment variable
# Install CUPTI included with CUDA by appending the LD_LIBRARY_PATH environment variable
export PATH="/usr/local/cuda-10.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"

# Install Anaconda, take Tsinghua mirror source and python 3.6.5 as an example
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.2.0-Linux-x86_64.sh
chmod a+x Anaconda3-5.2.0-Linux-x86_64.sh
./Anaconda3-5.2.0-Linux-x86_64.sh # Press f to turn the page, the default installation is in ~/anaconda directory, the installation process will prompt to modify the PATH variable

# Create a virtual environment
conda create -n wave-u-net python=3
conda activate wave-u-net

# Install dependencies
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch  # Pytorch 1.2.0 version has been tested
conda install tensorflow-gpu  # Only for tensorboard
conda install matplotlib
pip install tqdm librosa pystoi pesq

# Clone
git clone https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement.git
```

## Usage

There are two entry files for the current project:

- Entry file for training models: `train.py`
- Entry file for enhance noisy speech: `enhancement.py`

### Training

Use `train.py` to train the model. It receives three command line parameters:

- `-h`, display help information
- `-C, --config`, specify the configuration file required for training
- `-R, --resume`, continue training from the checkpoint of the last saved model

Syntax: `python train.py [-h] -C CONFIG [-R]`

E.g.:

```shell script
python train.py -C config/train/train.json
# The configuration file used to train the model is "config/train/train.json"
# Use all GPUs for training

python train.py -C config/train/train.json -R
# The configuration file used to train the model is "config/train/train.json"
# Use all GPUs to continue training from the last saved model checkpoint

CUDA_VISIBLE_DEVICES=1,2 python train.py -C config/train/train.json
# The configuration file used to train the model is "config/train/train.json"
# Use GPU No.1 and 2 for training

CUDA_VISIBLE_DEVICES=-1 python train.py -C config/train/train.json
# The configuration file used to train the model is "config/train/train.json"
# Use CPU for training
```

Supplement:

- Generally, the configuration files needed for training are placed in the `config/train` directory
- See the "Parameter Description" section for the parameters in the training configuration file
- The filename of the configuration file is the experiment name.

### Enhancement

Use `enhancement.py` to enhance noisy speech, which receives the following parameters:

- `-h, --help`, display help information
- `-C, --config`, specify the model, the enhanced dataset, and custom args used to enhance the speech.
- `-D, --device`, enhance the GPU index used, -1 means use CPU
- `-O, --output_dir`, specify where to store the enhanced speech, you need to ensure that this directory exists in advance
- `-M, --model_checkpoint_path`, the path of the model checkpoint, the extension of the checkpoint file is .tar or .pth

Syntax: `python enhancement.py [-h] -C CONFIG [-D DEVICE] -O OUTPUT_DIR -M MODEL_CHECKPOINT_PATH`

E.g.:

```shell script
python enhancement.py -C config/enhancement/unet_basic.json -D 0 -O enhanced -M /media/imucs/DataDisk/haoxiang/Experiment/Wave-U-Net-for-Speech-Enhancement/smooth_l1_loss/checkpoints/model_0020.pth
# The configuration file used to enhancement is "config/enhancement/unet_basic.json". Use this file to specify the model and dataset information required for enhancement
# Use GPU with index 0
# The output directory is "enhanced/", the directory needs to be created in advance
# Specify the path of the model checkpoint

python enhancement.py -C config/enhancement/unet_basic.json -D -1 -O enhanced -M /media/imucs/DataDisk/haoxiang/Experiment/Wave-U-Net-for-Speech-Enhancement/smooth_l1_loss/checkpoints/model_0020.tar
# Use CPU for enhancement
```

Supplement:

- Generally, the configuration files needed for enhancement are placed in the `config/enhancement/` directory
- See the "Parameter Description" section for the parameters in the enhancement configuration file.

## Visualization

All log information generated during training will be stored in the `config["root_dir"]/<config_filename>/` directory. Assuming that the configuration file for training is `config/train/sample_16384.json`, the value of the` root_dir` parameter in `sample_16384.json` is` /home/UNet/`. Then, the logs generated during the current experimental training process will be stored In the `/home/UNet/sample_16384/` directory. The directory will contain the following:

- `logs/` directory: store Tensorboard related data, including loss curve, waveform file, speech file
- `checkpoints/` directory: stores all checkpoints of the model, from which you can restart training or speech enhancement
- `config.json` file: backup of the training configuration file

During the training process, we can use `tensorboard` to start a static front-end server to visualize the log data in the relevant directory:

```shell script
tensorboard --logdir config["root_dir"]/<config_filename>/

# You can use --port to specify the port of the tensorboard static server
tensorboard --logdir config["root_dir"]/<config_filename>/ --port <port>

# For example, the "root_dir" parameter in the configuration file is "/home/happy/Experiments", the configuration file name is "train_config.json", and the default port is modified to 6000. The following commands can be used:
tensorboard --logdir /home/happy/Experiments/train_config --port 6000
```

## Directory description

During the training, multiple directories will be used, all with different purposes:

- Main directory: the directory where the current README.md is located, storing all source code
- Training directory: the `config["root_dir"]` directory in the training configuration file, which stores all experiment logs and model checkpoints of the current project
- Experiment directory: `config["root_dir"]/<experiment name>/` directory, which stores the log information of a certain experiment

## Parameter Description

### Training

`config/train/<config_filename>.json`

The log information generated during the training process will be stored in`config["root_dir"]/<config_filename>/`.

```json5
{
    "seed": 0, // Random seeds to ensure experiment repeatability
    "description": "...",  // Experiment description, will be displayed in Tensorboard later
    "root_dir": "~/Experiments/Wave-U-Net", // Directory for storing experiment results
    "cudnn_deterministic": false,
    "trainer": { // For training process
        "module": "trainer.trainer", // Which trainer
        "main": "Trainer", // The concrete class of the trainer model
        "epochs": 1200, // Upper limit of training
        "save_checkpoint_interval": 10, // Save model breakpoint interval
        "validation":{
        "interval": 10, // validation interval
         "find_max": true, // When find_max is true, if the calculated metric is the known maximum value, it will cache another copy of the current round of model checkpoint.
        "custon": {
            "visualize_audio_limit": 20, // The interval of visual audio during validation. The reason for setting this parameter is that visual audio is slow
            "visualize_waveform_limit": 20, // The interval of the visualization waveform during validation. The reason for setting this parameter is because the visualization waveform is slow
            "visualize_spectrogram_limit": 20, // Verify the interval of the visualization spectrogram. This parameter is set because the visualization spectrum is slow
            "sample_length": 16384 // See train dataset
            } 
        }
    },
    "model": {
        "module": "model.unet_basic", // Model files used for training
        "main": "Model", // Concrete class of training model
        "args": {} // Parameters passed to the model class
    },
    "loss_function": {
        "module": "model.loss", // Model file of loss function
        "main": "mse_loss", // Concrete class of loss function
        "args": {} // Parameters passed to the model class
    },
    "optimizer": {
        "lr": 0.001,
        "beta1": 0.9,
        "beat2": 0.009
    },
    "train_dataset": {
        "module": "dataset.waveform_dataset", // Store the training set model file
        "main": "Dataset", // Concrete class of training dataset
        "args": { // The parameters passed to the training set class, see the specific training set class for details
            "dataset": "~/Datasets/SEGAN_Dataset/train_dataset.txt",
            "limit": null,
            "offset": 0,
            "sample_length": 16384,
            "mode":"train"
        }
    },
    "validation_dataset": {
        "module": "dataset.waveform_dataset",
        "main": "Dataset",
        "args": {
            "dataset": "~/Datasets/SEGAN_Dataset/test_dataset.txt",
            "limit": 400,
            "offset": 0,
            "mode":"validation"
        }
    },
    "train_dataloader": {
        "batch_size": 120,
        "num_workers": 40, // How many threads to start to preprocess the data
        "shuffle": true,
        "pin_memory":true
    }
}
```

### Enhancement

`config/enhancement/*.json`

```json5
{
    "model": {
        "module": "model.unet_basic",  // Store the model file
        "main": "UNet",  // The specific model class in the file
        "args": {}  // Parameters passed to the model class
    },
    "dataset": {
        "module": "dataset.waveform_dataset_enhancement",  // Store the enhancement dataset file
        "main": "WaveformDataset",  // Concrete class of enhacnement dataset
        "args": {  // The parameters passed to the dataset class, see the specific enhancement dataset class for details
            "dataset": "/home/imucs/tmp/UNet_and_Inpainting/data.txt",
            "limit": 400,
            "offset": 0,
            "sample_length": 16384
        }
    },
    "custom": {
        "sample_length": 16384
    }
}
```

During the enhancement, only the path of the noisy speech can be listed in the *.txt file, similar to this:

```text
# enhancement_*.txt

/home/imucs/tmp/UNet_and_Inpainting/0001_babble_-7dB_Clean.wav
/home/imucs/tmp/UNet_and_Inpainting/0001_babble_-7dB_Enhanced_Inpainting_200.wav
/home/imucs/tmp/UNet_and_Inpainting/0001_babble_-7dB_Enhanced_Inpainting_270.wav
/home/imucs/tmp/UNet_and_Inpainting/0001_babble_-7dB_Enhanced_UNet.wav
/home/imucs/tmp/UNet_and_Inpainting/0001_babble_-7dB_Mixture.wav
```

## TODO

- [x] Use full-length speech for validation
- [x] Enhancement script