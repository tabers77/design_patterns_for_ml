import numpy as np
import pandas as pd
from typing import Any
from conf.config import TrainerConfigs
import utils as ut


class Evaluator:
    """
    Class for evaluating regression models.

    Attributes:
        split_configs (SplitConfigs): Object containing split configurations.
        trainer_configs (Any): Trainer configurations for model training.
    """

    def __init__(self,  trainer_configs: TrainerConfigs):
        self.split_configs = trainer_configs.scorer.split_configs
        self.trainer_configs = trainer_configs

    def evaluate(self, model: Any, splits: Any) -> ut.DictResultHolder:
        """
        Evaluate the regression model using specified metrics.

        Parameters:
            model (Any): The regression model to be evaluated.
            splits (Any): Data splits for evaluation.

        Returns:
            utils.DictResultHolder: Object containing evaluation results.
        """

        scoring_metrics = self.trainer_configs.scorer.get_scoring_funcs()

        if self.split_configs.split_policy == 'x_y_splits_only':
            container = dict()

            for eval_metric_name, _ in scoring_metrics.items():
                eval_metric_name_fixed = '_'.join(eval_metric_name.split('_')[1:])
                container[eval_metric_name_fixed] = - round(np.mean(model['test_' + eval_metric_name]), 2)

            results_table = pd.DataFrame(container, index=[0])

            results_table['model_name'] = model['name']

            container['results_table'] = results_table
            return ut.DictResultHolder(container)

        predictions = model.predict(splits.x_test)
        container = dict()

        for eval_metric_name, eval_metric in scoring_metrics.items():
            if 'neg' in eval_metric_name:
                eval_metric_name = '_'.join(eval_metric_name.split('_')[1:])

            container[eval_metric_name] = round(eval_metric(splits.y_test, predictions), 2)

        results_table = pd.DataFrame(container, index=[0])

        results_table['model_name'] = model.name
        container['results_table'] = results_table

        return ut.DictResultHolder(container)
