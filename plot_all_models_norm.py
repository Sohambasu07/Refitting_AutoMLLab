import json
import numpy as np
import plotly.graph_objects as go

# Data from the JSON files
# datasets = [
#     "aps_failure", "spambase", "puma32H", "kr-vs-kp", "electricity",
#     "delta_ailerons", "cpu_small", "bank32nh", "abalone", "2dplanes"
# ]

run_path = "Runs/Run_20240207_003612/"

metric = "all_models_refit_scores_roc_auc"

models = [
    "LightGBMXT",
    "WeightedEnsemble_L2",
    "LightGBM",
    "LightGBMLarge",
    "CatBoost",
    "XGBoost",
    "RandomForestEntr",
    "NeuralNetTorch",
    "ExtraTreesEntr",
    "ExtraTreesGini",
    "RandomForestGini",
    "NeuralNetFastAI",

]

unused_models = ["KNeighborsDist", "KNeighborsUnif"]

with open(run_path+"exp_meta.json", 'r') as file:
    data = json.load(file)
    split_rs, vs, datasets = data["split_random_state"], data["val_split"], data["datasets"]

def extract_score_from_json(file_path, model):
    with open(file_path, 'r') as file:
        data = json.load(file)
        model_full = model + "_FULL"
        if model.__contains__("KNeigh") and file_path.__contains__("kr-vs-kp"):
            return np.nan, np.nan, data["n_samples"]
        return data[metric][model], data[metric][model_full], data["n_samples"]

####################


# for model in models:
#     fit_score, refit_score, size = extract_score_from_json(file_path, model)
#     fit_scores.append(fit_score)
#     refit_scores.append(refit_score)
#     sizes.append(size)


#####################

sizes = []
datasizes = []
best_val_scores = []
refit_val_scores = []

for i, dataset in enumerate(datasets):
    fit_scores, refit_scores = [], []
    file_path = run_path + dataset + "/train_meta.json"
    for model in models:
        fit_score, refit_score, size = extract_score_from_json(file_path, model)
        fit_scores.append(fit_score)
        refit_scores.append(refit_score)
    datasizes.append(str(size) + "<br>(" + dataset + ")")
    best_val_scores.append(fit_scores)
    refit_val_scores.append(refit_scores)
    sizes.append(size)

sizes, refit_val_scores, best_val_scores, datasizes, datasets= map(list, zip(*sorted(zip(sizes, refit_val_scores, best_val_scores, datasizes, datasets))))

from normalization import run_min_max_norm

norm_fit_score, norm_refit_score = run_min_max_norm(best_val_scores, refit_val_scores)


best_val_scores = list(map(list, zip(*norm_fit_score)))
refit_val_scores = list(map(list, zip(*norm_refit_score)))


# Save as JSON
# with open('my_data_fit_scores.json', 'w') as file:
#     json.dump(best_val_scores, file)
#
# with open('my_data_refit_scores.json', 'w') as file:
#     json.dump(refit_val_scores, file)


fig = go.Figure()
# Adding Fit Validation roc_auc
for idx, scores in enumerate(best_val_scores):
    fig.add_trace(go.Scatter(x=datasizes, y=scores, mode='lines+markers', name=f'Fit {models[idx]}'))
# Adding Refit Validation roc_auc
for idx, scores in enumerate(refit_val_scores):
    fig.add_trace(
        go.Scatter(x=datasizes, y=scores, mode='lines+markers', name=f'Refit {models[idx]}', line=dict(dash='dot')))
fig.update_xaxes(type='category')
# Update layout
fig.update_layout(
    title='Comparison of Fit and Refit Validation score (min-max normalized roc_auc) for each Dataset',
    xaxis_title='Dataset',
    yaxis_title='normalized roc_auc',
    legend_title='Model',
    margin=dict(l=20, r=20, t=40, b=20),
    paper_bgcolor="LightSteelBlue",
)
fig.update_layout(yaxis_range = [0, 1])
fig.show()
print()
# Check if datasets and scores lists are of the same length
# if len(datasets) != len(best_val_scores) or len(datasets) != len(refit_val_scores):
#     print("Error: The number of datasets and the number of scores must be the same.")
# else:
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(datasets, best_val_scores, label='Fit Validation roc_auc', marker='o')
#     plt.plot(datasets, refit_val_scores, label='Refit Validation roc_auc', marker='x')
#     plt.xlabel('dataset size')
#     plt.ylabel('roc_auc')
#     plt.title('Comparison of Fit and Refit Validation ROC_AUC Across Datasets in order of dataset size (vs='+str(vs)+', random_state='+str(split_rs)+')')
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()