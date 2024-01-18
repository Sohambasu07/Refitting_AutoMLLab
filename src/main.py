from __future__ import annotations
import yaml
from pathlib import Path
import os
import argparse

from src.fit_gluon import Gluon

def main(dataset_dir: str,
         configs_dir: str,
         dataset_cfg: str,
         root_dir: Path = Path(os.getcwd()),
         mode: str = 'train',
         ):
    
    dataset_dir = root_dir / dataset_dir
    configs_dir = root_dir / configs_dir
    dataset_cfg = configs_dir / dataset_cfg

    with open(dataset_cfg, 'r') as f:
        datasets_config = yaml.safe_load(f)

    datasets = list(datasets_config.keys())

    gluon = Gluon(data_dir = dataset_dir,
                  datasets_config = datasets_config
                  )
    
    if mode == 'train':
        gluon.fit_gluon()


if __name__ == "__main__":
    mode_choices = ['train', 'eval']

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
                        default="train")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    
    main(dataset_dir = args.dataset_dir,
         configs_dir = args.configs_dir,
         dataset_cfg = args.dataset_cfg,
         root_dir = root_dir,
         mode = args.mode,
         )
    
    

    