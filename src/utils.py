from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io.arff import loadarff


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
    fig, ax = plt.subplots()
    results.plot.bar(ax = ax)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy of AutoML on synthetic datasets")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()