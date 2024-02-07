import pandas as pd
import json
import os

# Placeholder for the list of file paths. You should populate this list with the actual paths to your JSON files.
file_paths = ['Runs/Run_20240207_003612/spambase/train_meta.json']  # Add the rest of your file paths here

# Assuming the metric to compare is 'roc_auc' and is located under a key like 'all_models_refit_scores_roc_auc'
metric = 'all_models_refit_scores_roc_auc'
run_path = "Runs/Run_20240207_003612/"
with open(run_path+"exp_meta.json", 'r') as file:
    data = json.load(file)
    datasets = data["datasets"]
models = [
    "LightGBMXT",
    "LightGBM",
    "LightGBMLarge",
    "CatBoost",
    "XGBoost",
    "RandomForestEntr",
    "NeuralNetTorch",
    "ExtraTreesEntr",
    "ExtraTreesGini",
    "RandomForestGini",
    "NeuralNetFastAI"
]

def extract_data_and_create_label(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        rows = []
        if metric in data:
            for model, scores in data[metric].items():
                if not model.endswith('_FULL') and f"{model}_FULL" in data[metric] and 'model_complexity' in data and not model.__contains__('L2') and not model.__contains__('KNeigh'):
                    fit_score = scores
                    refit_score = data[metric][f"{model}_FULL"]
                    n_samples = data['n_samples']
                    n_features = data['n_features']
                    over_balanced = data['over_balanced']
                    under_balanced = data['under_balanced']
                    model_complexity = data['model_complexity'][model]
                    label = 1 if refit_score > fit_score else 0
                    rows.append([n_samples, n_features, over_balanced, under_balanced, model_complexity, label])
        return rows

# Initialize a list to collect all rows
all_rows = []

# Iterate over each file path and extract data
for dataset in datasets:
    all_rows.extend(extract_data_and_create_label(run_path+dataset+'/train_meta.json'))

# Create the DataFrame
df = pd.DataFrame(all_rows, columns=['n_samples', 'n_features', 'over_balanced', 'under_balanced', 'model_complexity', 'label'])

# Display the first few rows of the DataFrame to verify its structure
print(df.head())
csv_file_path = 'meta_learning_model_data.csv'
df.to_csv(csv_file_path, index=False)