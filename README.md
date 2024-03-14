<div align="center">

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python-3776AB?&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.13-EE4C2C?logo=pytorch&logoColor=white"></a>
<a href="https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html"><img alt="Tensorboard" src="https://img.shields.io/badge/Logging-Tensorboard-FF6F00"></a>
<a href="https://optuna.org"><img alt="Optuna" src="https://img.shields.io/badge/Optuna-integrated-blue"></a>
<a href="https://app.clear.ml"><img alt="Signup" src="https://img.shields.io/badge/Clear%7CML-Signup-brightgreen"></a>
<a href="https://black.readthedocs.io/en/stable"><img alt="L: Hydra" src="https://img.shields.io/badge/Code Style-Black-black" ></a>
</div>

This repository contains a fully Automated Benchmarking System For Medical Imaging Segmentation, which facilitates the comparison of state-of-the-art models' performance.
Our system enables architecture selection from a diverse range of models and automates the optimization of loss functions and parameter tuning through different search space algorithms. It also integrates a centralized logging system that tracks and shares all experiment-related details during training in real-time dashboards.
we have explored various model architectures encompassing different families, including the [nnU-Net](https://arxiv.org/abs/1809.10486), which automatically adapts to the dataset fingerprint, the [AttentionU-Net](https://arxiv.org/abs/1804.03999) utilizes an attention mechanism for feature extraction, and [UNETR](https://arxiv.org/abs/2103.10504).
Using our system enabled us to systematically fine-tune hyperparameters and investigate different loss functions, such as region-based losses like Tversky, and compound losses like Dice Cross Entropy. We also assessed both Adam and SGD optimizers, utilizing their respective update strategies. Additionally, we employed schedulers like Adaptive  and Plateau, which dynamically adjust the learning rate. To reduce computation time, we employed data downsampling techniques.

The following contains information about how to [set up the data](#setting-up-the-data).
A comparison between different SOTA approaches (Unet,nnU-Net,AttentionU-Net,UNETR) on the Lits and
AbdomenCT datasets is shown in the [experiments](#experiments) section.

## Table of Contents
- [Overview](#overview)
    - [References](#references)
- [Repository Structure](#repository-structure)
- [How To Run](#how-to-run)
  - [Requirements](#requirements)
  - [Setting up the Data](#setting-up-the-data)
    - [Lits](#lits)
    - [Selecting a Model](#selecting-a-model)
    - [Selecting a Dataset](#selecting-a-dataset)
    - [Changing Hyperparmeters](#changing-hyperparmeters)
    - [Logging and Checkpointing](#logging-and-checkpointing)
      - [Start using Tensorboard](#start-using-tensorboard)
      - [Start using ClearML](#start-using-clearml)
- [Experiments](#experiments)
    - [Defining the BaseModel](#defining-the-basemodel)
      - [Number of Epochs and Batch Size](#number-of-epochs-and-batch-size)
    - [Different Models](#different-models)
    - [Different Loss Functions](#different-loss-functions)
    - [Different Optimizer](#different-optimizer)
    - [Different Schedulers](#different-schedulers)
    - [Time Complexity](#time-complexity)
  - [AbdomenCT](#abdomenct)

# Overview

Overview about the results on the **Lits** Dataset.
The best result from runs (Dice score) is reported.
A more detailed analysis is given in the [experiments](#defining-the-basemodel) section.

| Model                | Dice     |
|----------------------|:--------:|
| UNET                 |  0.9574  |
| NnU-Net              |  0.9645  |
| AttentionU-Net       |  0.9599  |
| UNETR                |  0.9356  | 

### References

This repository adopts code from the following sources:

- **UNet** ([paper](https://arxiv.org/pdf/1505.04597.pdf), [source code](https://docs.monai.io/en/stable/networks.html#unet))
- **nnU-Net** ([paper](https://arxiv.org/abs/1809.10486), [source code](https://github.com/MIC-DKFZ/nnUNet))
- **AttentionU-Net** ([paper](https://arxiv.org/abs/1804.03999), [source code](https://docs.monai.io/en/stable/networks.html#attentionunet))
- **UNETR** ([paper](https://arxiv.org/abs/2103.10504), [source code](https://docs.monai.io/en/stable/networks.html#unetr))
  
# Repository Structure
```
Repository Standard_LiverLesion_Segmentation
│   README.md
│   requirements.txt    
│   environment.yml
│   .gitignore
|   .pre-commit-config.yaml
└─── experimental
│   │   Overlay.ipynb
│   │   Slice Paths Generator.ipynb
│   │   cropping_dataset.ipynb
│   │   main.ipynb
│   
└─── liver_imaging_analysis
|   │   config
|   |   │   configs.json
|   |   engine
|   |   │   config.py
|   |   │   engine.py
|   |   │   dataloader.py
|   |   │   transforms.py
|   |   │   utils.py
|   |   |   visualization.py
|   |   models
|   |   │   lesion_segmentation.py
|   |   │   liver_segmentation.py
|   |   │   lobe_segmentation.py
|   |   │   spleen_segmentation.py
│       
└─── tests
|   │  test_config.py
|   │  test_engine.py
|   │  test_models.py
|   │  test_transforms.py
|   │  test_utils.py
|   │  test_visualization.py
|   |  test_dataloader.py
│   
└─── scripts

```

# How To Run

## Requirements

Install the needed packages by the following command. You might need to adapt the cuda versions for
torch and torchvision specified in *requirements.txt*.
Find a pytorch installation guide for your
system [here](https://pytorch.org/get-started/locally/#start-locally).

````shell
pip install -r requirements.txt
````

Among others, this repository is mainly built on the following packages.
You may want to familiarize yourself with their basic use beforehand.

- **[Pytorch](https://pytorch.org/)**: The machine learning/deep learning library used in this
  repository.
- **[Black](https://black.readthedocs.io/en/stable/)**: Code Formatter used in this Repo. Check out the [official docs](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html) on how to use Black or [here](https://akshay-jain.medium.com/pycharm-black-with-formatting-on-auto-save-4797972cf5de) on how to integrate Black into PyCharm.

## Setting up the Data

Follow the instructions below to set up the respective datasets.

### Lits

<details><summary>Click to expand/collapse</summary>
<p>

Download the Lits dataset from [here](https://competitions.codalab.org/competitions/17094#participate).
Put them into a folder, the structure of the folder should now look like this:

````
LiverLesion_Segmentation(Lits)
|
└───OriginalData
│   │
│   └───Training_Data
│   |   │   segmentation-0.nii
│   |   │   segmentation-1.nii
│   |   │   ...
│   |   │   volume-0.nii
│   |   │   volume-1.nii
│   |   │   ...
│   |
│   └───Test_Data
│       │   test-volume-0.nii
│       │   test-volume-1.nii
│       │   ...

````

LiTS is a liver tumor segmentation benchmark. The data and segmentations are provided by various clinical sites around the world. The training data set contains 130 CT scans and the test data set 70 CT scans.

</p>
</details>

### Selecting a Model

You can change the model. The default model is Res-UNet.
**Available options for 'model' are: nnU-Net, AttentionU-Net,UNETR**.


### Selecting a Dataset

In the same way as the model, also the dataset can be changed. The default dataset is Lits.
**Available options for 'dataset' are: Lits, AbdomenCT**.

### Changing Hyperparmeters

Our Benchmarking System allowed us to systematically fine-tune hyperparameters and explore various loss functions, including region-based losses like Tversky, Dice and Focal Loss as well as compound losses like Dice Cross Entropy. We also evaluated the performance of both Adam and SGD optimizers, utilizing their respective update strategies. Furthermore, we incorporated schedulers such as AdaptiveLR and Plateau, which dynamically adjust the learning rate based on the training progress.

### Logging and Checkpointing
#### [Start using Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html#run-tensorboard)

The logging structure of the output folder is depicted below.
The ``LOGDIR=<some.folder.dir>`` argument defines the logging folder (*"logs/"* by default).
For a better overview, experiments can also be named by ``name="my_experiment"`` ("basemodel"
by default).
The parameters which are logged are: Hyperparameters (like epochs, batch_size, initial lr,...), 
metrics, loss (train+val), time (train+val) and learning rate.

````
LOGDIR                       # logs/ by default
    └──checkpoints/          # If checkpointing is enabled this contains the best and the last epochs checkpoint
    ├── logs/                # Console logging file
    └── events/              # Tensorboard logs      
````

Since [Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html#run-tensorboard)
is used for logging, you can view the logged results by the following command.
Tensorboard will include all logs (*.tfevents.* files) found in any subdirectory of *--logdir*.
This means by the level of the selected *--logdir* you can define which experiments (runs) should be
included into the tensorboard session.

````shell
tensorboard --logdir=<some.dir>
````

#### [Start using ClearML](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps) 

The ClearML run-time components:

* The ClearML Python Package - for integrating ClearML into scripts by adding just two lines of code, and optionally extending experiments and other workflows with ClearML's powerful and versatile set of classes and methods.
* The ClearML Server - for storing experiment, model, and workflow data; supporting the Web UI experiment manager and MLOps automation for reproducibility and tuning. It is available as a hosted service and open source to deploy ClearML Server.

<img src="https://raw.githubusercontent.com/allegroai/clearml-docs/main/docs/img/clearml_architecture.png" width="100%" alt="clearml-architecture">

1. Sign up for free to the [ClearML Hosted Service](https://app.clear.ml) (alternatively, you can set up your own server, see [here](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server)).

    > **_ClearML Demo Server:_** ClearML no longer uses the demo server by default. To enable the demo server, set the `CLEARML_NO_DEFAULT_SERVER=0`
    > environment variable. Credentials aren't needed, but experiments launched to the demo server are public, so make sure not 
    > to launch sensitive experiments if using the demo server.

2. Install the `clearml` python package:

    ```bash
    pip install clearml
    ```

3. Connect the ClearML SDK to the server by [creating credentials](https://app.clear.ml/settings/workspace-configuration), then execute the command
below and follow the instructions: 

    ```bash
    clearml-init
    ```

1. Add two lines to your code:
    ```python
    from clearml import Task
    task = Task.init(project_name='examples', task_name='hello world')
    ```

Everything your process outputs is now automagically logged into ClearML. 

# Experiments
The following experiments were performed under the following training settings and the reported
results are for the Lits validation set.
Additionally, the batch size is set to 8 and the number of epochs to 100 (see [Defining the Basemodel](#defining_the_basemodel)).

### Defining the BaseModel

This trains Res-UNet on the Lits Dataset with the following default settings:
Adam Optimizer with LR 0.01,scheduler StepLR 
and Four residual blocks.
Dice Loss  is used.

#### Number of Epochs and Batch Size

Looking at further hyperparameters, it can be seen that the batch size in particular is critical.
With the number of epochs, the fluctuation is much smaller, but a suitable value is still important.
Resulting from the experiments, a batch size of 8 and 100 epochs are used for further experiments.

### Different Models

After defining the basemodel on Res-UNet, other models are also trained and validated under same
conditions.

<details><summary>Appendix</summary>
<p>

| Model                | Dice     |
|----------------------|:--------:|
| UNET                 |  0.9574  |
| NnU-Net              |  0.9645  |
| AttentionU-Net       |  0.9599  |
| UNETR                |  0.9356  |

</p>
</details>

### Different Loss Functions

Looking at the different loss functions, it can be seen that the best results can be achieved with
Focal Loss.

<details><summary>Appendix</summary>
<p>

| Model | Experiment | mean mIoU |   
|:-----:|:----------:|:---------:|
| UNET  | Dice Loss  |   0.9574  |
| UNET  | Focal Loss |   0.9652  | 
| UNET  |   Tversky  |   0.81    |
| UNET  |Generalized |   0.9583  |  
| UNET  |   DC+CE    |   0.9613  | 

</p>
</details>

### Different Optimizer

Looking at the different optimizers, it can be seen that the best results can be achieved with
SGD.

<details><summary>Appendix</summary>
<p>

| Model | Experiment | mean mIoU |   
|:-----:|:----------:|:---------:|
| UNET  |    Adam    |   0.9574  |
| UNET  |    SGD     |   0.9669  | 
| UNET  |  RMSProp   |   0.95    |

</p>
</details>

### Different Schedulers

Looking at the different Schedulers, it can be seen that the best results can be achieved with
StepLR.

<details><summary>Appendix</summary>
<p>

| Model | Experiment | mean mIoU |   
|:-----:|:----------:|:---------:|
| UNET  |   StepLR   |   0.9574  |
| UNET  |  Plateau   |   0.8968  | 
| UNET  |  Cyclic    |   0.9019  |


</p>
</details>

### Time Complexity

The following figure shows the training and inference time of each model, as well as the number of
parameters.
It can be seen that the number of parameters is in a similar range for all models, but still as the
number of parameters increases, the time complexity of the models also increases.

![Time Complexity](     "Time Complexity")

## AbdomenCT

Some experiments are repeated using another dataset, which is the AbdomenCT dataset.
AbdomenCT-1K, with more than 1000 (1K) CT scans from 12 medical centers, including multi-phase, multi-vendor, and multi-disease cases.

<details><summary>Appendix</summary>
<p>

| Model | Experiment |   Dice    |   
|:-----:|:----------:|:---------:|
| UNET  | AbdomenCT  |   95.3    |


</p>
</details>