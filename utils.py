from conf.config import Cfg
from typing import Any, Optional, Dict


class Scorers:
    """
    Class for managing scoring functions used in model evaluation.

    Attributes:
        split_configs (SplitConfigs): Object containing split configurations.
        custom_scoring (Optional[Any]): Custom scoring functions.
    """

    def __init__(self, split_configs: Any, custom_scoring: Optional[Any] = None):
        """
        Initialize the Scorers.

        Parameters:
            split_configs (SplitConfigs): Object containing split configurations.
            custom_scoring (Optional[Any]): Custom scoring functions. Defaults to None.
        """
        self.split_configs = split_configs
        self.custom_scoring = custom_scoring

    def get_scoring_funcs(self) -> Any:
        """
        Get the scoring functions based on the split policy and custom scoring.

        Returns:
            Any: Scoring functions to be used for evaluation.
        """
        scoring_funcs = (
            self.custom_scoring
            if self.custom_scoring is not None
            else (
                Cfg.scoring_funcs.regression_scoring_funcs_cv
                if self.split_configs.split_policy == 'x_y_splits_only'
                else Cfg.scoring_funcs.regression_scoring_funcs
            )
        )

        return scoring_funcs


class DictResultHolder:
    """
    Class for holding evaluation results in a dictionary format.

    This class is designed to hold evaluation results, which can be of various types and structures,
    in a dictionary format for easy access and manipulation.

    Attributes:
        results (Dict[str, Any]): A dictionary containing evaluation results.
    """

    def __init__(self, results: Dict[str, Any]):
        """
        Initialize the DictResultHolder with a dictionary of evaluation results.

        Parameters:
            results (Dict[str, Any]): A dictionary containing evaluation results.
        """
        for key, value in results.items():
            setattr(self, key, value)
