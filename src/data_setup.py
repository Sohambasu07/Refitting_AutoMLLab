import os
from pathlib import Path
import requests
import yaml
import argparse

def download_datasets(data_dir, 
                      dataset_name, 
                      url,
                      endswith):
    if (dataset_name + endswith) in os.listdir(data_dir):
        print(f"Dataset {dataset_name} already downloaded!")
        return
    
    print(f"Downloading dataset {dataset_name}...")
    r = requests.get(url, verify = False)
    with open(data_dir / (dataset_name + endswith), "wb") as f:
        f.write(r.content)
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=os.getcwd())
    parser.add_argument("--dataset_cfg", type=str, default="configs/datasets.yaml")
    parser.add_argument("--data_dir", type=str, default="data")

    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    data_dir = root_dir / args.data_dir
    if not data_dir.exists():
        data_dir.mkdir()
    dataset_cfg = root_dir / args.dataset_cfg

    with open(dataset_cfg, "r") as f:
        dataset_config = yaml.safe_load(f)

    # print(list(dataset_config.keys()))

    for dataset in dataset_config:
        print(dataset)
        # print(dataset_config[dataset])
        download_datasets(data_dir = data_dir,
                          dataset_name = dataset,
                          url = dataset_config[dataset]["url"],
                          endswith = ".arff"
                          )
