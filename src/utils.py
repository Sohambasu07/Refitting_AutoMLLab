from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io.arff import loadarff
import yaml
import os
import json
import argparse
import pprint


def arff_to_dataframe( 
                data_dir: Path,
                dataset: str) -> pd.DataFrame:
        """ Convert .arff to Pandas dataframe """
        print(f"Loading dataset {dataset}...")
        data = loadarff(data_dir / (dataset + '.arff'))
        df = pd.DataFrame(data[0])
        return df

def plot_results(
        results: pd.DataFrame,
        save_path: str | None = None
) -> None:
        """ Plot the results of the experiments """
        print("Plotting the results...")
        plt.figure(figsize=(10, 5))
        plt.plot(results.index, results['best_fit_score'], label='best_fit_score', marker='^')
        plt.plot(results.index, results['best_val_score'], label='best_val_score', marker='o')
        plt.xticks(results.index)
        plt.legend()
        plt.show()
        if save_path is not None:
                plt.savefig(save_path)
                plt.show()

def agg_results(
        dir_list: list[str],
        datasets: list[str],
        list_val_splits: list[float],
        main_run_dir: str = 'Runs',
        root_dir: Path = Path(os.getcwd()),
):
        """ Aggregate the results of the experiments """

        print("Aggregating the results...")

        main_run_dir = root_dir / main_run_dir
        run_info = dict()

        for i, run_dir in enumerate(dir_list):
                run_info[list_val_splits[i+1]] = dict()
                for dataset in datasets:
                        with open(main_run_dir / run_dir / dataset / 'train_meta.json', 'r') as f:
                                train_meta = json.load(f)
                                run_info[list_val_splits[i+1]][dataset] = {
                                        'best_fit_score': abs(train_meta['best_fit_score']),
                                        'best_val_score': abs(train_meta['best_val_score'][train_meta['eval_metric']]),
                                        # 'refit_val_score': train_meta['refit_val_score']['log_loss']
                                        }
                                
        # pp = pprint.PrettyPrinter(indent=2)
        # pp.pprint(run_info)
        
        for dataset in datasets:
                results = pd.DataFrame(columns = ['best_fit_score', 'best_val_score', 'refit_val_score'])
                for val_split in list_val_splits[1:]:
                        results.loc[val_split] = run_info[val_split][dataset]
                results.index.name = 'val_split'
                results = results.sort_index()
                print(results)
                plot_results(
                        results = results,
                        save_path = None # main_run_dir / 'agg_results' / (dataset + '.png')
                )
        



if __name__ == "__main__":

        parser = argparse.ArgumentParser()

        parser.add_argument("--root_dir", "-r",
                        type=Path, 
                        default=os.getcwd())
        
        parser.add_argument("--main_run_dir", "-m",
                        type=str, 
                        default='Runs')
        
        parser.add_argument("--config_dir", "-c",
                        type=str, 
                        default='configs')
        
        parser.add_argument("--exp_config", "-ec",
                        type=str, 
                        default='exp_config.yaml')
        
        parser.add_argument("--exp_name", "-en",
                        type=str, 
                        default="Exp1")
        
        args = parser.parse_args()

        configs_dir = args.root_dir / args.config_dir
        exp_config_path = configs_dir / args.exp_config
        main_run_dir = args.root_dir / args.main_run_dir

        with open(exp_config_path, 'r') as f:
                exp_config = yaml.safe_load(f)

        dir_list = exp_config['Experiments'][args.exp_name]['dir_list']
        datasets = exp_config['Experiments'][args.exp_name]['datasets']

        print(dir_list)

        list_val_splits = set()
        for run_dir in dir_list:
                with open(main_run_dir / run_dir / 'exp_meta.json', 'r') as f:
                        config = json.load(f)
                        list_val_splits.add(config['val_split'])        
        if len(list_val_splits) != len(exp_config['Experiments'][args.exp_name]['dir_list']):
                raise ValueError("No two experiments must have the same validation split!")
        
        list_val_splits = list(list_val_splits)
        list_val_splits.insert(0, 0.0)
        
        agg_results(
                dir_list = dir_list,
                datasets = datasets,
                list_val_splits = list_val_splits,
                main_run_dir = main_run_dir,
                root_dir = args.root_dir
        )