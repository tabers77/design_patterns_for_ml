import numpy as np
import pandas as pd
import sklearn.metrics as m
from typing import Dict, Any
from conf.config import Cfg


class Evaluator:
    """
    Class for evaluating regression models using specified metrics.
    """

    def __init__(self, split_configs):
        self.split_configs = split_configs

    def evaluate(self, model, splits) -> 'RegressionResults':

        if self.split_configs.split_policy == 'x_y_splits_only':
            container = dict()

            for eval_metric_name, _ in Cfg.scoring_funcs.regression_scoring_funcs_cv.items():

                eval_metric_name_fixed = '_'.join(eval_metric_name.split('_')[1:])

                container[eval_metric_name_fixed] = - round(np.mean(model['test_' + eval_metric_name]), 2)

            results_table = pd.DataFrame(container, index=[0])
            container['results_table'] = results_table
            return RegressionResults(container)

        predictions = model.predict(splits.x_test)
        container = dict()
        for eval_metric_name, eval_metric in Cfg.scoring_funcs.regression_scoring_funcs.items():
            container[eval_metric_name] = round(eval_metric(splits.y_test, predictions), 2)

        results_table = pd.DataFrame(container, index=[0])
        results_table['model_name'] = model.name
        container['results_table'] = results_table

        return RegressionResults(container)


class RegressionResults:
    """
    Class for holding regression evaluation results.
    """

    def __init__(self, results: Dict[str, Any]):
        """
        Initialize the RegressionResults with a dictionary of evaluation results.

        Parameters:
        - results: A dictionary containing evaluation results.
        """
        for eval_metric, result in results.items():
            setattr(self, eval_metric, result)
