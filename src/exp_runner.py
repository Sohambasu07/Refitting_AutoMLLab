from __future__ import annotations

from typing import Dict, Tuple, Any
import pandas as pd
from scipy.io.arff import loadarff
import os
import datetime
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

    labels: list[str]
    """ List of labels for each dataset """

    predictors: list[Gluon]
    """ List of predictors for each dataset """

    holdout_frac: float
    """ Fraction of the dataset to be used as holdout for validation """

    eval_metric: str
    """ Evaluation metric to be used for training """

    root_dir: Path
    """ Root directory """

    def __init__(
            self,
            data_dir: Path,
            dataset_config: Dict[str, Any],
            spec_dataset: list[str] | None = None,
            holdout_frac: float = 0.1,
            eval_metric: str = 'accuracy',
            root_dir: Path = Path(os.getcwd())
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
        self.holdout_frac = holdout_frac
        self.eval_metric = eval_metric
        self.root_dir = root_dir


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

    def run_exp(
            self,
            mode: str,
            refit_dir: str | None = None
    ):
        
        if mode == 'fit':
            if os.path.exists(self.root_dir / 'Runs') is False:
                os.mkdir(self.root_dir / 'Runs')

            run_path = self.root_dir / 'Runs' / f'Run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.mkdir(run_path)

            print("Running experiment in mode 'fit'...")
            for i, df in enumerate(self.dataframes):
                label = self.labels[i] if len(self.labels) > 1 else self.labels[0]
                print("\n\n!!!!!!!!!!!!!!!===========================!!!!!!!!!!!!!!!!")
                print(f"DATASET {i}")
                print(f"Fitting predictor for dataset {self.datasets[i]}...")
                self.predictors.append(Gluon.fit_gluon(
                    dataframe = df, 
                    label = label,
                    eval_metric = self.eval_metric,
                    holdout_frac = self.holdout_frac,
                    save_path = run_path / f'{self.datasets[i]}'
                    ))
        
        elif mode == 'refit':

            if refit_dir is None:
                raise ValueError("No refit directory specified!")
            
            run_path = self.root_dir / 'Runs'
            if os.path.exists(run_path) is False:
                raise ValueError(f"No Runs directory found in {self.root_dir}!"
                                 "Please run the experiment in mode 'fit' first!")
            
            refit_path = run_path / refit_dir
            print(refit_path)
            if os.path.exists(refit_path) is False:
                raise ValueError(f"Refit directory {refit_dir} not found in {run_path}!")
            
            for dataset in self.datasets:
                print(dataset)
                if dataset not in os.listdir(refit_path):
                    continue
                self.predictors.append(
                    Gluon.load_predictor(
                    path = refit_path / f'{dataset}'
                    ))

            if len(self.predictors) == 0:
                raise ValueError("No predictors to refit!]"
                                "Please run the experiment in mode 'fit' first!")
            
            print("Running experiment in mode 'refit'...")
            for i, predictor in enumerate(self.predictors):
                print("\n\n!!!!!!!!!!!!!!!===========================!!!!!!!!!!!!!!!!")
                print(f"DATASET {i}")
                print(f"Refitting predictor for dataset {self.datasets[i]}...")
                self.predictors[i] = Gluon.refit_gluon(
                    predictor = predictor
                    )

