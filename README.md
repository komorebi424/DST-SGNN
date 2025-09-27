# DST-SGNN

The full paper and appendix are available at [Arxiv](https://arxiv.org/abs/2506.00798).

## Appendix
The appendix of the paper can be accessed through [Appendix](https://github.com/komorebi424/DST-SGNN/blob/master/Appendix.pdf). This appendix contains several supplementary materials, such as detailed verification processes, experimental data, supplementary analyses, or other information that supports the main content of the paper but is not included in the main text due to space limitations. 

## Overview

DST-SGNN: Dynamic Spatio-Temporal Stiefel Graph Neural Network

This project introduces the Dynamic Spatio-Temporal Stiefel Graph Neural Network (DST-SGNN), designed to efficiently forecast spatio-temporal time series (STTS) data. STTS appear in various applications, but forecasting them is challenging due to complex dependencies across both time and space.

LAD-SGNN: LLM-Augmented Stiefel Graph Neural Networks

LAD-SGNN is an LLM-enhanced version of DST-SGNN (IJCAI) for spatiotemporal forecasting. Building on Stiefel spectral graph convolution, LAD-SGNN incorporates large language model features to improve both structural and semantic modeling.

Key improvements and highlights:

Comprehensive review and context: Model design is informed by a thorough reorganization of spatiotemporal forecasting studies, including recent LLM-based approaches.

LLM integration with structured prompts: A lightweight LLM framework is embedded with a spatio-temporal alignment mechanism, enabling unified structural and semantic modeling.

Expanded experimental evaluation: Evaluated on zero-shot and few-shot forecasting across ten real-world datasets, supported by ablation and sensitivity analyses.

This repository provides code and examples to replicate LAD-SGNN experiments and explore its LLM-augmented forecasting capabilities.

This repository provides code and examples to replicate LAD-SGNN experiments and explore its LLM-augmented forecasting capabilities.
*(Note: Files with `_L` in their names and the scripts `RUN_FEW.py` and `RUN_ZERO.py` are related to **LAD-SGNN**. Specifically, `RUN_FEW.py` and `RUN_ZERO.py` are used for few-shot and zero-shot experiments, and their usage is similar to **DST-SGNN**.)* 

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

