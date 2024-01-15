import os
from pathlib import Path
import requests

class Dataset_Setup:
    def __init__(self, 
                 data_dir: str, 
                 url: str, 
                 dataset_name: str, 
                 root_dir: Path = Path(os.getcwd())):
        self.root_dir = root_dir
        self.data_dir = root_dir / data_dir
        self.url = url
        self.dataset_name = dataset_name

    def download(self):
        if os.path.exists(self.data_dir / self.dataset_name):
            print("Dataset already downloaded!")
            return
        
        print("Downloading dataset...")
        r = requests.get(self.url)
        endswith = self.url.split(".")[-1]
        with open(self.data_dir / self.dataset_name + endswith, "wb") as f:
            f.write(r.content)
        print("Download complete.")
