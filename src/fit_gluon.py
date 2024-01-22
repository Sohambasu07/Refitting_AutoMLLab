from __future__ import annotations
from typing import Dict, Tuple, TYPE_CHECKING, Any
from autogluon.tabular import TabularPredictor

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame

class Gluon:

    def fit_gluon(dataframe: DataFrame,
                  label: str
                  ) -> TabularPredictor:
        predictor = TabularPredictor(label = label).fit(dataframe)
        return predictor

    def refit_gluon(predictor: TabularPredictor
                    ) -> TabularPredictor:
        predictor.refit_full()
        return predictor

    # def evaluate_gluon(self, test_data):
    #     self.predictor.evaluate(test_data)