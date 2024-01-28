import pandas as pd
import sklearn.metrics as metrics
from typing import Dict, Any


class Evaluator:
    """
    Class for evaluating regression models using specified metrics.
    """

    def __init__(self):
        self.regression_metrics: Dict[str, Any] = {'mean_squared_error': metrics.mean_squared_error,
                                                   'mean_absolute_error': metrics.mean_absolute_error,
                                                   'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error,

                                                   }

    def evaluate(self, model, x_test, y_test) -> 'RegressionResults':
        predictions = model.predict(x_test)
        container = dict()
        for eval_metric_name, eval_metric in self.regression_metrics.items():
            container[eval_metric_name] = eval_metric(y_test, predictions)

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
