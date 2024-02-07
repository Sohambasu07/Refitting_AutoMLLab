from autogluon.tabular import TabularPredictor
import pandas as pd
# Load the DataFrame from the saved CSV file
# (This step could be skipped if you just created the DataFrame, but is shown here for completeness)
df_loaded = pd.read_csv("meta_learning_model_data.csv")

# Splitting the DataFrame into features and target label
label = 'label'
features = df_loaded.drop(columns=[label])
target = df_loaded[label]


# AutoGluon TabularPredictor requires no separate feature, target split. 
# It identifies the target based on the label column name provided.
predictor = TabularPredictor(label=label, eval_metric='accuracy').fit(df_loaded)

training_performance = predictor.evaluate(df_loaded)

print("Training performance (error):", training_performance)