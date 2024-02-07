# To Refit or Not to Refit? Automating Design Decisions for AutoML Systems 

## Overview
This project explores the question of whether it's beneficial to refit the final model in an AutoML system. Often, during the validation procedure, a portion of the data is held out from training, potentially leading to suboptimal model performance. Refitting the final model on all available data could potentially improve performance, but it's challenging to determine beforehand if refitting is necessary. This project aims to provide heuristics for determining when refitting is beneficial.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)

## Introduction

AutoML systems automate the process of selecting and optimizing machine learning models for a given dataset. One crucial decision in the AutoML pipeline is whether to refit the selected model on the entire training dataset after validation. This decision can significantly impact the final model's performance and generalization ability.

In this project, we investigate the effects of refitting on the performance of AutoML models across various datasets. We propose a method for automatically determining whether to refit the model based on validation scores and dataset characteristics.

### Research Question

How can we determine if we should refit the final model in an AutoML system?

### Method Used

The project aims to find heuristics to determine when refitting the final model in an AutoML system is necessary. By exploring various factors and experimental setups, we seek to provide insights into when refitting can lead to improved model performance.


## Setup

To use the code in this repository, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/your-repository.git
```

2. Install the required dependencies listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```

3. Download datasets:
```python
python src/data_setup.py --root_dir /path/to/root
```

## Usage

Make sure to set up the necessary configurations in the script, such as the file paths, dataset names, and model names.

Run Experiments: Use the main.py script to run experiments. Specify the desired mode (fit, refit, eval, plot, info) and provide any necessary parameters via command-line arguments.

- For fit:
```python
python main.py --mode fit --spec_dataset dataset1 dataset2 --holdout_frac 0.2 --val_split 0.1
```

- For refit:
```python
python main.py --mode refit --dir Run_YYYYMMDD_HHMMSS
```

View Experiments Results:
This will generate an interactive plot displaying the ROC-AUC scores for each model across different datasets.
```python
python plot_all_models_norm.py
```