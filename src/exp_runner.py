from __future__ import annotations

from typing import Dict, Tuple, Any
import pandas as pd
import os
import datetime
from pathlib import Path

from src.fit_gluon import Gluon
from src.utils import arff_to_dataframe

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
                
            df = arff_to_dataframe(
                data_dir = self.data_dir, 
                dataset = dataset)
            
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
            verbosity: int = 2,
            evaluate: bool = False,
            eval_dir: str | None = None,
            test_data: Dict[str, Any] | None = None,
            refit_dir: str | None = None
    ) -> None:
        """ 
        Run the experiment in the specified mode 

        Modes:
        ----------

        if mode == 'fit':
            args required: None
            args optional: test_data
        
        if mode == 'refit':
            args required: refit_dir
            args optional: None

        if mode == 'eval':
            args required: eval_dir, test_data
            args optional: None

        Parameters:
        -----------

        mode: str
            Mode to run the experiment in
            Options: 'fit', 'refit', 'eval'

        verbosity: int
            Verbosity level for the predictor
            Only applicable for mode == 'fit'
            Options: 1, 2, 3

        evaluate: bool
            Whether to evaluate the predictor on a separate test dataset during training
            Only applicable for mode == 'fit'

        eval_dir: str
            Directory to load the predictors from for evaluation
            Only applicable for mode == 'eval'

        test_data: Dict[str, Any]
            Dictionary containing the test dataframes and corresponding label key
            Only applicable for mode == 'fit' and 'eval'

        refit_dir: str
            Directory to load the predictors from for refitting
            Only applicable for mode == 'refit'

        Returns:
        --------
        None

        """


        # MODE: FIT

        if mode == 'fit':
        
            if os.path.exists(self.root_dir / 'Runs') is False:
                os.mkdir(self.root_dir / 'Runs')
            
            run_path = self.root_dir / 'Runs' / f'Run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.mkdir(run_path)
            
            print("Running experiment in mode 'fit'...")
            # Train and save the predictors
            for i, df in enumerate(self.dataframes):
                label = self.labels[i]
                print("\n\n!!!!!!!!!!!!!!!===========================!!!!!!!!!!!!!!!!")
                print(f"DATASET {i}")
                print(f"Fitting predictor for dataset {self.datasets[i]}...")
                self.predictors.append(Gluon.fit_gluon(
                    dataframe = df, 
                    label = label,
                    eval_metric = self.eval_metric,
                    holdout_frac = self.holdout_frac,
                    save_path = run_path / f'{self.datasets[i]}',
                    verbosity = verbosity
                    ))
                
                if evaluate:
                    if test_data is None:
                        raise ValueError("No test dataset provided!")
                # Evaluate the predictor on the test dataset
                    test_df = test_data['data']
                    test_label = test_data['label']
                    test_df.rename(columns = {test_label: label}, inplace = True)
                    print("Evaluating the predictor on the test dataset...")
                    score = Gluon.evaluate_gluon(
                        test_dataframe = test_df,
                        predictor = self.predictors[i]
                        )
                    print(f"Test score for predictor of dataset {self.datasets[i]}: {score}")


        # MODE: REFIT
        
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
            

            # Load the predictors
            for dataset in self.datasets:
                print(dataset)

                # Skip datasets that are not in the refit directory
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


        # MODE: EVAL
        
        elif mode == 'eval':
            if eval_dir is None:
                raise ValueError("No Eval directory specified!")
            
            run_path = self.root_dir / 'Runs'
            if os.path.exists(run_path) is False:
                raise ValueError(f"No Runs directory found in {self.root_dir}!"
                                "Please run the experiment in mode 'fit' first!")
            
            eval_path = run_path / eval_dir
            print(eval_path)
            if os.path.exists(eval_path) is False:
                raise ValueError(f"Eval directory {eval_dir} not found in {run_path}!")
            
            if test_data is None:
                raise ValueError("No test dataset provided!")
            
            # Load the predictors
            for dataset in self.datasets:
                print(dataset)

                # Skip datasets that are not in the eval directory
                if dataset not in os.listdir(eval_path):
                    continue
                self.predictors.append(
                    Gluon.load_predictor(
                    path = eval_path / f'{dataset}'
                    ))
                self.labels.append(self.dataset_config[dataset]['label'])
            
            if len(self.predictors) == 0:
                raise ValueError("No predictors to evaluate!]"
                                "Please run the experiment in mode 'fit' first!")
            
            print("Running experiment in mode 'eval'...")
            for i, predictor in enumerate(self.predictors):
                label = self.labels[i]
                test_df = test_data['data']
                test_label = test_data['label']
                test_df.rename(columns = {test_label: label}, inplace = True)
                print("\n\n!!!!!!!!!!!!!!!===========================!!!!!!!!!!!!!!!!")
                print(f"DATASET {i}")
                print(f"Evaluating predictor for dataset {self.datasets[i]}...")
                score = Gluon.evaluate_gluon(
                        test_dataframe = test_df,
                        predictor = self.predictors[i]
                        )
                print(f"Test score for predictor of dataset {self.datasets[i]}: {score}")

        elif mode == 'plot':
            pass

        else:
            raise ValueError(f"Invalid mode {mode}!")

