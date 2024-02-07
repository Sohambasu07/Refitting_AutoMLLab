# To Refit or Not to Refit? Automating Design Decisions for AutoML Systems 

## Overview
This project explores the question of whether it's beneficial to refit the final model in an AutoML system. Often, during the validation procedure, a portion of the data is held out from training, potentially leading to suboptimal model performance. Refitting the final model on all available data could potentially improve performance, but it's challenging to determine beforehand if refitting is necessary. This project aims to provide heuristics for determining when refitting is beneficial.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Meta-Learning for Decision-Making](#meta-learning-for-decision-making)
- [Supervisors](#supervisors)

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

## Meta-Learning for Decision-Making

In this project, we employ meta-learning techniques to assist us in making informed decisions based on the results of our experiments. Meta-learning enables our system to learn from previous experiments and adapt its decision-making process accordingly. By analyzing various metrics and factors such as dataset characteristics, model complexity, and performance scores, our meta-learning model helps us determine whether to refit or not, ultimately enhancing the overall effectiveness of our machine learning pipeline.

### Key Components

- **Data Extraction**: We extract relevant data from our experiment results, including performance metrics, dataset properties, and model complexity measures.
  
- **Label Creation**: Based on the extracted data, we create labels indicating whether it is beneficial to refit the model or not, leveraging meta-learning principles to make these determinations.
  
- **Model Training**: Utilizing AutoGluon, we train a meta-learning model on the generated dataset, enabling it to learn patterns and relationships from past experiments.

- **Decision Support**: The trained meta-learning model provides valuable insights and recommendations for decision-making, helping us optimize our machine learning workflows and achieve better performance.

### Usage

To utilize our meta-learning approach for decision-making:

1. Ensure you have collected experiment data in the required format, including performance metrics and dataset characteristics.
2. Run the provided script to process the experiment data and create a meta-learning dataset.
3. Train the meta-learning model using AutoGluon on the generated dataset.
4. Utilize the trained model to analyze new experiment results and make informed decisions on whether to refit models based on the learned patterns.

By incorporating meta-learning into our workflow, we enhance our ability to make effective decisions and adapt to varying conditions encountered in machine learning experimentation.

## Supervisors

We would like to express our gratitude to the following supervisors for their guidance and support throughout the project:

- [Lennart Purucker](#https://ml.informatik.uni-freiburg.de/profile/purucker/)
- [Eddie Bergman](#https://ml.informatik.uni-freiburg.de/profile/bergman/)