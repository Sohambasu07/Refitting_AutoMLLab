from __future__ import annotations

from typing import Dict, Tuple, TYPE_CHECKING, Any
import pandas as pd
from scipy.io.arff import loadarff

if TYPE_CHECKING:
    from pathlib import Path

from src.fit_gluon import Gluon


class ExpRunner:
    data_dir: Path
    """ Path to the directory containing the dataset(s) """

    dataset_config: Dict[str, Any]
    """ Dictionary containing the dataset(s) configurations """

    spec_dataset: list[str] | None
    """ Specified dataset(s) or no. of datasets to use
        If None: all datasets will be used
        If list[str]: the specified dataset(s) will be used"""

    datasets: list[str]
    """ List of dataset names """

    dataframes: list[pd.DataFrame]
    """ List of dataframes containing the dataset(s) """

    def __init__(self,
                 data_dir: Path,
                 dataset_config: Dict[str, Any],
                 spec_dataset: list[str] | None = None
                 ):
        self.data_dir = data_dir
        self.dataset_config = dataset_config
        self.spec_dataset = spec_dataset
        if isinstance(spec_dataset, list):
            self.datasets = spec_dataset
        else:
            self.datasets = list(dataset_config.keys())
        self.dataframes, self.labels = self.load_data()
        self.predictors = []

    def load_data(self) -> Tuple[list[pd.DataFrame], list[str]]:
        dataframes = []
        labels = []
        for dataset in self.dataset_config:
            if isinstance(self.spec_dataset, list):
                if dataset not in self.spec_dataset:
                    continue
            print(f"Loading dataset {dataset}...")
            data = loadarff(self.data_dir / (dataset + '.arff'))
            df = pd.DataFrame(data[0])
            dataframes.append(df)
            labels.append(self.dataset_config[dataset]['label'])
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

    def run_exp(self,
                mode: str
                ):
        
        if mode == 'fit':
            print("Running experiment in mode 'fit'...")
            for i, df in enumerate(self.dataframes):
                label = self.labels[i] if len(self.labels) > 1 else self.labels[0]
                print("\n\n!!!!!!!!!!!!!!!===========================!!!!!!!!!!!!!!!!")
                print(f"DATASET {i}")
                print(f"Fitting predictor for dataset {self.datasets[i]}...")
                self.predictors.append(Gluon.fit_gluon(dataframe = df, 
                                                       label = label))
        
        elif mode == 'refit':
            print("Running experiment in mode 'refit'...")
            for i, df in enumerate(self.dataframes):
                print("\n\n!!!!!!!!!!!!!!!===========================!!!!!!!!!!!!!!!!")
                print(f"DATASET {i}")
                print(f"Refitting predictor for dataset {self.datasets[i]}...")
                self.predictors.append(Gluon.refit_gluon(predictor = self.predictors[i]))

