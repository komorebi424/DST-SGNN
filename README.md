# DST-SGNN

The full paper and appendix are available at [Arxiv](https://arxiv.org/abs/2506.00798).

## Appendix
The appendix of the paper can be accessed through [Appendix](https://github.com/komorebi424/DST-SGNN/blob/master/Appendix.pdf). This appendix contains several supplementary materials, such as detailed verification processes, experimental data, supplementary analyses, or other information that supports the main content of the paper but is not included in the main text due to space limitations. 

## Overview

This project introduces **Dynamic Spatio-Temporal Stiefel Graph Neural Network (DST-SGNN)**, designed to efficiently forecast spatio-temporal time series (STTS) data. STTS are widely used in various applications, but their forecasting is challenging due to complex dependencies in both time and space dimensions.

Files with `_L` in their names are related to **LDSGNN**.

## Prerequisites
- Python 3.10.14
- PyTorch 2.1.2
- torchvision 0.16.2
- torchaudio 2.1.2
- PyTorch-CUDA 12.1

## Setup Instructions

1. **Install Python 3.10.14** (if you haven't already).
2. **Create and activate a new conda environment** (optional but recommended):
    ```bash
    conda create -n your_env_name python=3.10.14
    conda activate your_env_name
    ```

3. **Install the required libraries**:
    Run the following command to install the necessary dependencies:
    ```bash
    conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```


## Usage

1. **Place your dataset** in the `data` folder.
2. **Run the `RUN.py` script**:
    Open the terminal in VSCode and run the following command:
    ```bash
    python RUN.py
    ```

## Hyperparameters

The hyperparameters are set via command-line arguments or the `hyperparameter.txt` file.

### Parameters:
- `--data`: Dataset (e.g., `PEMS03`)
- `--feature_size`: Feature size (e.g., `358`)
- `--batch_size`: Batch size (e.g., `2`)
- `--train_epochs`: Epochs (e.g., `25`)
- `--seq_length`: Sequence length (e.g., `336`)
- `--pre_length`: Forecast length (e.g., `96`)

### Usage:
You can pass parameters directly:
```bash
python RUN.py --data PEMS03 --feature_size 358 --batch_size 2 --train_epochs 25 --seq_length 336 --pre_length 96
```
Or：
Modify `RUN.py`, then run:
```bash
python RUN.py

