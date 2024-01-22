from __future__ import annotations
from typing import Dict, Tuple, TYPE_CHECKING, Any
from autogluon.tabular import TabularPredictor
import torch

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame

class Gluon:

    def fit_gluon(
            dataframe: DataFrame,
            label: str,
            eval_metric: str = 'accuracy',
            holdout_frac: float = 0.1,
            save_path: str = None
    ) -> TabularPredictor:
        predictor = TabularPredictor(
            label = label,
            eval_metric = eval_metric,
            path = save_path
            ).fit(
                dataframe,
                holdout_frac = holdout_frac,
                num_gpus = torch.cuda.device_count()
                )
        return predictor

    def refit_gluon(
            predictor: TabularPredictor,
    ) -> TabularPredictor:
        predictor.refit_full('all')
        return predictor

    # def evaluate_gluon(self, test_data):
    #     self.predictor.evaluate(test_data)