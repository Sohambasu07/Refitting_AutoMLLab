from __future__ import annotations

from typing import Dict, Tuple, Any
import pandas as pd
import os
import datetime
from pathlib import Path
import json

from src.fit_gluon import Gluon
from src.utils import arff_to_dataframe
from sklearn.model_selection import train_test_split

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

    holdout_frac: float | None
    """ Fraction of the dataset to be used as holdout for validation """

    val_split: float | None
    """ Fraction of the dataset to be used as validation split outside AutoGluon"""

    split_random_state: int | None
    """ Random state for the train test split """

    eval_metric: str
    """ Evaluation metric to be used for training """

    root_dir: Path
    """ Root directory """

    presets: list[str] | None
    """ List of presets to be used for training """

    def __init__(
            self,
            data_dir: Path,
            dataset_config: Dict[str, Any],
            spec_dataset: list[str] | None = None,
            holdout_frac: float | None = None,
            val_split: float | None = None,
            split_random_state: int | None = None,
            eval_metric: str = 'accuracy',
            root_dir: Path = Path(os.getcwd()),
            presets: list[str] | None = None
    ):
        self.data_dir = data_dir
        self.dataset_config = dataset_config
        self.spec_dataset = spec_dataset
        if isinstance(spec_dataset, list):
            self.datasets = spec_dataset
        else:
            self.datasets = list(dataset_config.keys())
        self.dataframes = []
        self.labels = []
        self.val_dataframes = []
        self.val_labels = []
        self.predictors = []
        self.holdout_frac = holdout_frac
        self.val_split = val_split
        self.eval_metric = eval_metric
        self.root_dir = root_dir
        self.presets = presets
        self.split_random_state = split_random_state if split_random_state is not None else 42


    def load_data(self) -> Tuple[list[pd.DataFrame], list[str]]:
        dataframes = []
        labels = []
        for dataset in self.dataset_config:
            if dataset not in self.datasets:
                continue
            df = arff_to_dataframe(
                data_dir = self.data_dir, 
                dataset = dataset)
            
            dataframes.append(df)
            labels.append(self.dataset_config[dataset]['label'])
        return dataframes, labels
    
    def split_data(self) -> None:
        if self.val_split is None or self.val_split == 0:
            return
        for i, df in enumerate(self.dataframes):
            label = self.labels[i]
            df_train, df_val = train_test_split(
                df,
                test_size = self.val_split,
                random_state = self.split_random_state,
            )
            self.dataframes[i] = df_train
            self.val_dataframes.append(df_val)
            self.val_labels.append(label)
    
    
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
            directory: str | None = None,
            test_data: Dict[str, Any] | None = None,
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

        directory: str
            Directory to load the predictors from for refitting, 
            evaluation, and info
            Only applicable for mode == 'refit', 'eval', 'info'

        test_data: Dict[str, Any]
            Dictionary containing the test dataframes and 
            corresponding label key
            Only applicable for mode == 'fit' and 'eval'

        Returns:
        --------
        None

        """


        # MODE: FIT

        if mode == 'fit':

            # Loading the dataset(s) and labels
            self.dataframes, self.labels = self.load_data()
            
            # Splitting the dataset(s) into train and validation sets
            self.split_data()
        
            if os.path.exists(self.root_dir / 'Runs') is False:
                os.mkdir(self.root_dir / 'Runs')
            
            run_path = self.root_dir / 'Runs' / f'Run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.mkdir(run_path)

            # Save the experiment metadata

            exp_meta = {
                'datasets': self.datasets,
                'num_datasets': len(self.datasets),
                'holdout_frac': self.holdout_frac,
                'val_split': self.val_split,
                'split_random_state': self.split_random_state,
                'eval_metric': self.eval_metric,
                'presets': self.presets,
                'refit': False
            }

            with open(run_path / 'exp_meta.json', 'w') as f:
                json.dump(exp_meta, f, indent = 4)
            
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
                    verbosity = verbosity,
                    presets = self.presets
                    ))
                score = None
                if self.val_split is not None:
                # Evaluate the predictor on the validation set
                    print("Evaluating the predictor on the external validation split...")
                    score = Gluon.evaluate_gluon(
                        test_dataframe = self.val_dataframes[i],
                        predictor = self.predictors[i]
                        )
                    print(f"Test score for predictor of dataset {self.datasets[i]}: {score}")

                ft_sum = self.predictors[i].leaderboard(silent = True)
                all_fit_scores = ft_sum['score_val'].tolist()
                all_models = ft_sum['model'].tolist()

                dataset_class_percent = df[label].value_counts(normalize=True)
                over_balanced_percent = dataset_class_percent.max()
                under_balanced_percent = dataset_class_percent.min()

                # Save the train metadata
                train_meta = {
                'dataset': self.datasets[i],
                'label': label,
                'n_samples': df.shape[0] + self.val_dataframes[i].shape[0],
                'n_features': df.shape[1],
                'n_classes': len(df[label].unique()),
                'over_balanced': over_balanced_percent,
                'under_balanced': under_balanced_percent,
                'holdout_frac': self.holdout_frac,
                'validation_split': self.val_split,
                'eval_metric': self.eval_metric,
                'best_model': self.predictors[i].model_best,
                'best_fit_score': all_fit_scores[0],
                'best_val_score': score,
                'all_models_fit_scores': dict(zip(all_models, all_fit_scores)),
                'refit': False,
                'refit_val_score': None
                }

                with open(run_path / self.datasets[i] / 'train_meta.json', 'w') as f:
                    json.dump(train_meta, f, indent = 4)


        # MODE: REFIT
        
        elif mode == 'refit':

            # Run checks
            self.redundant_checks(
                mode = mode,
                directory = directory
            )

            # Load the experiment metadata
            with open(self.root_dir / 'Runs' / directory / 'exp_meta.json', 'r') as f:
                exp_meta = json.load(f)
                self.split_random_state = exp_meta['split_random_state']
                self.val_split = exp_meta['val_split']
                self.datasets = exp_meta['datasets']

            # Set the refit flag to True in the experiment metadata
            exp_meta['refit'] = True
            with open(self.root_dir / 'Runs' / directory / 'exp_meta.json', 'w') as f:
                json.dump(exp_meta, f, indent = 4)
            
            # Loading the dataset(s) and labels
            self.dataframes, self.labels = self.load_data()            
            
            # Load the predictors
            self.load_predictors_labels(
                directory = directory
            )
            
            # Train-val splitting according to fit's random state
            self.split_data()
            
            # Refitting the predictors
            print("Running experiment in mode 'refit'...")
            for i, predictor in enumerate(self.predictors):
                print("\n\n!!!!!!!!!!!!!!!===========================!!!!!!!!!!!!!!!!")
                print(f"DATASET {i}")
                print(f"Refitting predictor for dataset {self.datasets[i]}...")
                self.predictors[i] = Gluon.refit_gluon(
                    predictor = predictor
                    )
                
                score = None
                ft_sum = None
                if self.val_split is not None:
                # Evaluate the predictor on the validation set
                    print("Evaluating the refitted predictor on the external validation split...")
                    score = Gluon.evaluate_gluon(
                        test_dataframe = self.val_dataframes[i],
                        predictor = self.predictors[i]
                        )
                    ft_sum = self.predictors[i].leaderboard(data=self.val_dataframes[i],
                                                            extra_metrics=['roc_auc', 'log_loss'],
                                                            silent=True)
                    print(f"Test score for refitted predictor of dataset {self.datasets[i]}: {score}")

                all_refit_scores_roc_auc = ft_sum['roc_auc'].tolist()
                all_refit_scores_log_loss = ft_sum['log_loss'].tolist()
                all_models = ft_sum['model'].tolist()

                # Set the refit flag to True and save the refit score in the train metadata
                with open(self.root_dir / 'Runs' / directory / self.datasets[i] / 'train_meta.json', 'r') as f:
                    train_meta = json.load(f)
                    train_meta['refit'] = True
                    train_meta['refit_val_score'] = score
                    train_meta['model_complexity'] = self.calculate_complexity(self.predictors[i])
                    train_meta['all_models_refit_scores_roc_auc'] = dict(zip(all_models, all_refit_scores_roc_auc))
                    train_meta['all_models_refit_scores_log_loss'] = dict(zip(all_models, all_refit_scores_log_loss))

                with open(self.root_dir / 'Runs' / directory / self.datasets[i] / 'train_meta.json', 'w') as f:
                    json.dump(train_meta, f, indent = 4)


        # MODE: EVAL
        
        elif mode == 'eval':
            
            # Run checks
            self.redundant_checks(
                mode = mode,
                directory = directory,
                test_data = test_data
            )
            
            # Load the predictors
            self.load_predictors_labels(
                directory = directory
            )
            
            # Evaluating the predictors
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

        elif mode == 'info':
            
            # Run checks
            self.redundant_checks(
                mode = mode,
                directory = directory
            )
            

            # Load the predictors
            self.load_predictors_labels(
                directory = directory
            )
            
            # Getting info for the predictors
            for i, predictor in enumerate(self.predictors):
                print("\n\n!!!!!!!!!!!!!!!===========================!!!!!!!!!!!!!!!!")
                print(f"DATASET {i}")
                print(f"Getting info for predictor of dataset {self.datasets[i]}...")
                info = Gluon.get_info(
                    predictor = predictor
                    )
                print(info)

        else:
            raise ValueError(f"Invalid mode {mode}!")
        
    def redundant_checks(
            self,
            mode: str,
            directory: str | None = None,
            test_data: Dict[str, Any] | None = None
    ):
        if directory is None:
                raise ValueError(f"No {mode} directory specified!")
            
        run_path = self.root_dir / 'Runs'
        if os.path.exists(run_path) is False:
            raise ValueError(f"No Runs directory found in {self.root_dir}!"
                            "Please run the experiment in mode 'fit' first!")
        
        dir_path = run_path / directory

        print(dir_path)
        if os.path.exists(dir_path) is False:
            raise ValueError(f"{mode} directory {dir_path} not found in {run_path}!")
        
        if mode == 'eval':
            if test_data is None:
                raise ValueError("No test dataset provided!")

    def load_predictors_labels(
            self,
            directory: str | None = None,
    ):
            
        run_path = self.root_dir / 'Runs'
        dir_path = run_path / directory
        # Load the predictors
        for dataset in self.datasets:
            print(dataset)

            # Skip datasets that are not in the eval directory
            if dataset not in os.listdir(dir_path):
                continue
            self.predictors.append(
                Gluon.load_predictor(
                path = dir_path / f'{dataset}'
                ))
            self.labels.append(self.dataset_config[dataset]['label'])

        if len(self.predictors) == 0:
            raise ValueError("No predictors present!]"
                            "Please run the experiment in mode 'fit' first!")

    def calculate_complexity(self,
                             predictor: Gluon):
        # Define weights for each parameter
        weight_epochs = 0.6
        weight_boost_iterations = 0.6
        weight_estimators = 0.6
        weight_fit_time = 0.3
        weight_pred_time = 0.1

        # Get the leaderboard with extra info
        feat_sm = predictor.leaderboard(extra_info=True)

        complexity_scores = {}

        for index, model_info in feat_sm.iterrows():
            if "FULL" in model_info['model'] or "WeightedEnsemble_L2" in model_info['model'] or "KNeighbors" in model_info['model']:
                continue
            
            num_epochs, num_boost_iterations, n_estimators = 0, 0, 0
            hyperparams_fit = model_info['hyperparameters_fit']
            if len(hyperparams_fit) > 0:
                if 'num_epochs' in hyperparams_fit:
                    num_epochs = hyperparams_fit['num_epochs']
                elif 'epochs' in hyperparams_fit:
                    num_epochs = hyperparams_fit['epochs']

                num_boost_iterations = max(hyperparams_fit.get('num_boost_round', 0), hyperparams_fit.get('iterations', 0))
                n_estimators = hyperparams_fit.get('n_estimators', n_estimators)

            fit_time = model_info['fit_time']
            pred_time = model_info['pred_time_val']
            num_epochs_norm = num_epochs / 1000
            num_boost_iterations_norm = num_boost_iterations / 10000
            n_estimators_norm = n_estimators / 10000

            # Calculate weighted complexity score
            complexity_score = (weight_epochs * num_epochs_norm +
                                weight_boost_iterations * num_boost_iterations_norm +
                                weight_estimators * n_estimators_norm +
                                weight_fit_time * fit_time +
                                weight_pred_time * pred_time)
            
            complexity_scores[model_info['model']] = complexity_score
        
        return complexity_scores