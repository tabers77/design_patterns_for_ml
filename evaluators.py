import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from conf.config import Cfg
from conf.config import SplitConfigs


class Evaluator:
    """
    Class for evaluating regression models using specified metrics.
    """

    def __init__(self, split_configs: SplitConfigs, custom_scoring: Optional[Dict[str, Any]] = None):
        self.split_configs = split_configs
        self.custom_scoring = custom_scoring

    def evaluate(self, model: Any, splits: Any) -> 'RegressionResults':
        """
        Evaluate the regression model using specified metrics.

        Parameters:
        - model: The regression model to be evaluated.
        - splits: Data splits for evaluation.

        Returns:
        - RegressionResults: Object containing evaluation results.
        """
        scoring_funcs = self.custom_scoring if self.custom_scoring is not None else \
            (Cfg.scoring_funcs.regression_scoring_funcs_cv if self.split_configs.split_policy == 'x_y_splits_only'
             else Cfg.scoring_funcs.regression_scoring_funcs)

        if self.split_configs.split_policy == 'x_y_splits_only':
            container = dict()

            for eval_metric_name, _ in scoring_funcs.items():
                eval_metric_name_fixed = '_'.join(eval_metric_name.split('_')[1:])
                container[eval_metric_name_fixed] = - round(np.mean(model['test_' + eval_metric_name]), 2)

            results_table = pd.DataFrame(container, index=[0])
            container['results_table'] = results_table
            return RegressionResults(container)

        predictions = model.predict(splits.x_test)
        container = dict()

        for eval_metric_name, eval_metric in scoring_funcs.items():
            if 'neg' in eval_metric_name:
                eval_metric_name = '_'.join(eval_metric_name.split('_')[1:])

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
