from __future__ import annotations
from typing import Dict, Tuple, TYPE_CHECKING, Any
import pandas as pd
from scipy.io.arff import loadarff
from autogluon.tabular import TabularPredictor
import yaml

if TYPE_CHECKING:
    from pathlib import Path

class Gluon:
    def __init__(self,
                 data_dir: Path,
                 dataset_config: Dict[str, Any]
                 ):
        self.data_dir = data_dir
        self.datasets = dataset_config

        self.dataframes, self.labels = self.load_data()

    def load_data(self) -> Tuple[list[pd.DataFrame], list[str]]:
        dataframes = []
        labels = []
        for dataset in self.datasets:
            print(f"Loading dataset {dataset}...")
            data = loadarff(self.data_dir / (dataset + '.arff'))
            df = pd.DataFrame(data[0])
            dataframes.append(df)
            labels.append(self.datasets[dataset]['label'])
        return dataframes, labels
    
    def check_data(self):
        for df in self.dataframes:
            print("===========================")
            print(df.info())
            print(df.head())
            print(df.shape)
            print(df.columns)
            print(df.dtypes)
            print(df.describe())

    def fit_gluon(self):
        self.predictors = []
        for i, df in enumerate(self.dataframes):
            label = self.labels[i] if len(self.labels) > 1 else self.labels[0]
            predictor = TabularPredictor(label = label).fit(df)
            self.predictors.append(predictor)

    # def evaluate_gluon(self, test_data):
    #     self.predictor.evaluate(test_data)