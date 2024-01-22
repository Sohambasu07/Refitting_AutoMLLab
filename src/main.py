from __future__ import annotations
import yaml
from pathlib import Path
import os
import argparse

from src.exp_runner import ExpRunner

def main(dataset_dir: str,
         configs_dir: str,
         dataset_cfg: str,
         root_dir: Path = Path(os.getcwd()),
         mode: str = 'fit',
         spec_dataset: list[str] | int | None = None
         ):
    
    # Setting up the paths
    dataset_dir = root_dir / dataset_dir
    configs_dir = root_dir / configs_dir
    dataset_cfg = configs_dir / dataset_cfg

    # Loading the dataset config
    with open(dataset_cfg, 'r') as f:
        dataset_config = yaml.safe_load(f)

    datasets = list(dataset_config.keys())

    # Verbosity for datasets
    if spec_dataset is None:
        print("Using all datasets...")

    elif isinstance(spec_dataset, list):

        # Check if spec_dataset is valid
        for ds in spec_dataset:
            if ds not in datasets:
                print(f"Dataset {ds} is not a valid dataset!")
    
        spec_dataset = list(filter(lambda x: x in datasets, spec_dataset))

        print(f"Using datasets: {spec_dataset}")
    
    # Creating the ExpRunner object
    exp = ExpRunner(data_dir = dataset_dir,
                  dataset_config = dataset_config,
                  spec_dataset = spec_dataset
                  )
    
    # Displaying information about the data
    # exp.check_data()

    # Running the experiment
    exp.run_exp(mode = mode)    
    


if __name__ == "__main__":
    mode_choices = ['fit', 'refit']

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", 
                        type=str, 
                        default=os.getcwd())
    
    parser.add_argument("--dataset_dir", 
                        type=str, 
                        default="data")
    
    parser.add_argument("--configs_dir", 
                        type=str, 
                        default="configs")
    
    parser.add_argument("--dataset_cfg", 
                        type=str, 
                        default="datasets.yaml")
    
    parser.add_argument("--mode", 
                        type=str, 
                        choices=mode_choices,
                        default="fit")
    
    parser.add_argument("--spec_dataset",
                        nargs='+',
                        type=str,
                        help="Dataset(s) to use"
                        "If None: all datasets will be used"
                        "If list[str]: the specified dataset(s) will be used",
                        default=None)
    
    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    main(dataset_dir = args.dataset_dir,
         configs_dir = args.configs_dir,
         dataset_cfg = args.dataset_cfg,
         root_dir = root_dir,
         mode = args.mode,
         spec_dataset = args.spec_dataset
         )
    
    