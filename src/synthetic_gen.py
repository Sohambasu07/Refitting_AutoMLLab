from __future__ import annotations

from typing import Dict, Tuple, Any

from sklearn.datasets import make_classification
import argparse
from pathlib import Path
import os

def synthetic_gen(
        n_samples: int = 1000,
        n_features: int = 20,
        n_informative: int = 2,
        n_redundant: int = 2,
        n_repeated: int = 0,
        n_classes: int = 2,
        class_sep: float = 1.0,
        shuffle: bool = True,
        random_state: int | None = None,
        write: bool = False,
        root_dir: Path = Path(os.getcwd()),
        data_dir: str = 'data',
        configs_dir: str = 'configs'
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Generate a synthetic dataset using sklearn.datasets.make_classification
    """

    print("Generating synthetic dataset...")

    X, y = make_classification(
        n_samples = n_samples,
        n_features = n_features,
        n_informative = n_informative,
        n_redundant = n_redundant,
        n_repeated = n_repeated,
        n_classes = n_classes,
        class_sep = class_sep,
        shuffle = shuffle,
        random_state = random_state
    )
    # dataset = {'X': X, 'y': y}
    # dataset_config = {'label': 'y'}

    print("Dataset generated!")

    if write:
        data_dir = Path(root_dir) / data_dir
        configs_dir = Path(root_dir) / configs_dir
        # if not data_dir.exists():
        #     os.mkdir(data_dir)
        # if not configs_dir.exists():
        #     os.mkdir(configs_dir)

        print(f"Writing the synthetic dataset to {data_dir}...")
        
        # Writing the test dataset config
        with open(configs_dir / 'test_dataset.yaml', 'w') as f:
            f.write('\"test_dataset\":\n')
            f.write('   \"label\": \"y\"\n')

        # Writing the test dataset
        with open(data_dir / 'test_dataset.arff', 'w') as f:
            f.write('% Synthetic Test dataset generated using sklearn.datasets.make_classification\n')
            f.write('@relation test_dataset\n')
            for i in range(n_features):
                f.write(f'@attribute x{i} numeric\n')
            f.write('@attribute y {0, 1}\n')
            f.write('@data\n')
            for i in range(n_samples):
                for j in range(n_features):
                    f.write(f'{X[i, j]},')
                f.write(f'{y[i]}\n')
        
        print("Dataset written!")

    # return dataset, dataset_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", 
                        type = Path, 
                        default = Path(os.getcwd()))
    
    parser.add_argument("--data_dir",
                        type = str,
                        default = 'data')
    
    parser.add_argument("--configs_dir",
                        type = str,
                        default = 'configs')
    
    parser.add_argument("--write",
                        type = bool,
                        default = False)

    parser.add_argument("--n_samples", 
                        type = int, 
                        default = 1000)
    
    parser.add_argument("--n_features",
                        type = int,
                        default = 20)
    
    parser.add_argument("--n_informative",
                        type = int,
                        default = 2)
    
    parser.add_argument("--n_redundant",
                        type = int,
                        default = 2)
    
    parser.add_argument("--n_repeated",
                        type = int,
                        default = 0)
    
    parser.add_argument("--n_classes",
                        type = int,
                        default = 2)
    
    parser.add_argument("--class_sep",
                        type = float,
                        default = 1.0)
    
    args = parser.parse_args()

    synthetic_gen(
        n_samples = args.n_samples,
        n_features = args.n_features,
        n_informative = args.n_informative,
        n_redundant = args.n_redundant,
        n_repeated = args.n_repeated,
        n_classes = args.n_classes,
        class_sep = args.class_sep,
        write = args.write,
        root_dir = args.root_dir,
        data_dir = args.data_dir,
        configs_dir = args.configs_dir
    )

