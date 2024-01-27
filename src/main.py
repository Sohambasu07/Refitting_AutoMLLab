from __future__ import annotations
import yaml
from pathlib import Path
import os
import argparse

from src.exp_runner import ExpRunner
# from src.utils import arff_to_dataframe

def main(
        dataset_dir: str,
        configs_dir: str,
        dataset_cfg: str,
        holdout_frac: float | None = None,
        val_split: float | None = None,
        split_random_state: int | None = None,
        test_dataset_name: str | None = None,
        root_dir: Path = Path(os.getcwd()),
        mode: str = 'fit',
        spec_dataset: list[str] | int | None = None,
        eval_metric: str = 'accuracy',
        directory: str | None = None,
        verbosity: int = 2,
        presets: list[str] | None = None
):
    
    # Setting up the paths
    dataset_dir = root_dir / dataset_dir
    configs_dir = root_dir / configs_dir
    dataset_cfg = configs_dir / dataset_cfg

    # Loading the train dataset config
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


    # # Loading the test dataset config
    # with open(configs_dir / 'test_dataset.yaml', 'r') as f:
    #     test_dataset_config = yaml.safe_load(f)

    # # Loading the test dataset
    # test_df = arff_to_dataframe(
    #     data_dir = dataset_dir,
    #     dataset = test_dataset_name
    # )

    # test_label = test_dataset_config[test_dataset_name]['label']

    # test_data = {
    #     'data': test_df,
    #     'label': test_label
    # }
        

    
    # Creating the ExpRunner object
    exp = ExpRunner(data_dir = dataset_dir,
                    dataset_config = dataset_config,
                    spec_dataset = spec_dataset,
                    holdout_frac = holdout_frac,
                    val_split = val_split,
                    eval_metric = eval_metric,
                    presets = presets,
                    split_random_state = split_random_state,
                    )
    
    # Displaying information about the data
    # exp.check_data()

    # Running the experiment
    exp.run_exp(mode = mode,
                verbosity = verbosity,
                directory = directory,
                # test_data = test_data
                )    
    


if __name__ == "__main__":
    mode_choices = ['fit', 'refit', 'eval', 'plot', 'info']
    metric_choices = ['accuracy', 'log_loss', 'roc_auc']

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", "-r",
                        type=Path, 
                        default=os.getcwd())
    
    parser.add_argument("--dataset_dir", "-d",
                        type=str, 
                        default="data")
    
    parser.add_argument("--configs_dir", "-cd",
                        type=str, 
                        default="configs")
    
    parser.add_argument("--dataset_cfg", "-cfg",
                        type=str, 
                        default="datasets.yaml")
    
    parser.add_argument("--mode", "-m",
                        type=str, 
                        choices=mode_choices,
                        default="fit")
    
    parser.add_argument("--spec_dataset", "-sd",
                        nargs='+',
                        type=str,
                        help="Dataset(s) to use"
                        "If None: all datasets will be used"
                        "If list[str]: the specified dataset(s) will be used",
                        default=None)
    
    parser.add_argument("--holdout_frac", "-hf",
                        type=float,
                        help="Heldout fraction for the 'fit' experiment",
                        default=None)
    
    parser.add_argument("--val_split", "-vs",
                        type=float,
                        help="Validation split outside AutoGluon for the 'fit' experiment",
                        default=None)
    
    parser.add_argument("--eval_metric", "-em",
                        type=str,
                        help="Evaluation metric for the 'fit' experiment",
                        choices=metric_choices,
                        default="accuracy")
    
    parser.add_argument("--directory", "-dir",
                        type=str,
                        help="Directory to load the predictor from for refitting,"
                        " evaluation, plotting, or info",
                        default=None)
    
    parser.add_argument("--evaluate", "-e",
                        action="store_true",
                        help="Evaluate the predictor on the test dataset",
                        default=False)
    
    parser.add_argument("--test_dataset_name", "-td",
                        type=str,
                        help="Name of the test dataset",
                        default=None)
    
    parser.add_argument("--verbosity", "-v",
                        type=int,
                        help="Verbosity level",
                        default=2)
    
    parser.add_argument("--presets", "-p",
                        nargs='+',
                        type=str,
                        help="Preset(s) to use for the 'fit' experiment",
                        default=None)
    
    parser.add_argument("--split_random_state", "-srs",
                        type=int,
                        help="Random state for the train-val split",
                        default=None)
    
    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    main(
        dataset_dir = args.dataset_dir,
        configs_dir = args.configs_dir,
        dataset_cfg = args.dataset_cfg,
        holdout_frac = args.holdout_frac,
        val_split = args.val_split,
        split_random_state = args.split_random_state,
        test_dataset_name = args.test_dataset_name,
        root_dir = root_dir,
        mode = args.mode,
        spec_dataset = args.spec_dataset,
        eval_metric = args.eval_metric,
        directory = args.directory,
        verbosity = args.verbosity,
        presets = args.presets
    )