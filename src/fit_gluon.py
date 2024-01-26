from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
from autogluon.tabular import TabularPredictor
import torch

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame

class Gluon:

    def load_predictor(
            path: str
    ) -> TabularPredictor:
        predictor = TabularPredictor.load(path)
        return predictor

    def fit_gluon(
            dataframe: DataFrame,
            label: str,
            eval_metric: str = 'accuracy',
            holdout_frac: float | None = None,
            save_path: str = None,
            verbosity: int = 2
    ) -> TabularPredictor:
        predictor = TabularPredictor(
            label = label,
            eval_metric = eval_metric,
            path = save_path
            ).fit(
                dataframe,
                holdout_frac = holdout_frac,
                num_gpus = torch.cuda.device_count(),
                verbosity = verbosity
                )
        return predictor

    def refit_gluon(
            predictor: TabularPredictor,
    ) -> TabularPredictor:
        predictor.refit_full('all')
        return predictor

    def evaluate_gluon(
            test_dataframe: DataFrame,
            predictor: TabularPredictor
    ) -> Tuple[float, float]:
        score = predictor.evaluate(data = test_dataframe)
        return score
    
    def get_info(
            predictor: TabularPredictor
    ) -> dict:
        info = predictor.fit_summary(show_plot = True)
        return info