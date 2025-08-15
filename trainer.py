from sklearn.model_selection import cross_validate
import logging
from conf import config as cfg
from typing import Any, Optional, Union, Dict

logging.basicConfig(level=logging.INFO)

class Trainer:
    """
    Trainer class for handling training and cross-validation of models.
    """

    def __init__(self, splits: Any, split_configs: Any, trainer_configs: Any, model: Optional[Any] = None,
                 pipe_model: Optional[Any] = None) -> None:
        self.splits = splits
        self.split_configs = split_configs
        self.trainer_configs = trainer_configs
        self.model = model
        self.pipe_model = pipe_model
        self.custom_scoring = self.trainer_configs.custom_scoring

    def base_train(self) -> Union[Any, Dict[str, Any]]:
        """
        Perform base training without neural network.

        Returns:
        - Union[Any, Dict[str, Any]]: Trained model or cross-validation results.
        """

        if self.split_configs.split_policy == 'feature_target':
            if self.trainer_configs.preprocess_strategy == 'pipeline':
                self.pipe_model.fit(self.splits.x_train, self.splits.y_train)
                self.pipe_model.name = self.model.name
                logging.info("Pipeline model trained successfully.")
                return self.pipe_model

            elif self.trainer_configs.preprocess_strategy == 'custom':
                self.model.fit(self.splits.x_train, self.splits.y_train)
                logging.info("Custom model trained successfully.")
                return self.model

        elif self.split_configs.split_policy == 'x_y_splits_only':

            scoring_metrics = list(self.custom_scoring.keys()) if self.custom_scoring is not None else \
                list(cfg.Cfg.scoring_funcs.regression_scoring_funcs_cv.keys())

            if self.trainer_configs.preprocess_strategy == 'pipeline':

                try:
                    cv_results = cross_validate(self.pipe_model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                                scoring=scoring_metrics)
                except Exception as e:
                    logging.error(f'Error during cross-validation: {e}')
                    raise ValueError(
                        f'If using pipeline & cross validation check that you use the correct scoring parameters, '
                        f'you are using:{self.custom_scoring} . Error {e}')

                self.pipe_model.name = self.model.name
                logging.info("Pipeline cross-validation completed successfully.")
                return cv_results

            elif self.trainer_configs.preprocess_strategy == 'custom':

                cv_results = cross_validate(self.model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                            scoring=scoring_metrics)
                logging.info("Custom cross-validation completed successfully.")
                return cv_results

        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.trainer_configs.preprocess_strategy}")

    def nn_train(self) -> Union[Any, Dict[str, Any]]:
        """
        Perform training for neural networks.

        Returns:
        - Union[Any, Dict[str, Any]]: Trained model or cross-validation results.
        """

        if self.split_configs.split_policy == 'feature_target':
            if self.trainer_configs.preprocess_strategy == 'pipeline':
                raise NotImplementedError

            elif self.trainer_configs.preprocess_strategy == 'custom':
                self.model.fit(self.splits.x_train,
                               self.splits.y_train,
                               epochs=50,
                               batch_size=32,
                               validation_data=(self.splits.x_test, self.splits.y_test), verbose=2)
                logging.info("Neural network trained successfully.")
                return self.model

        elif self.split_configs.split_policy == 'x_y_splits_only':

            scoring_metrics = list(self.custom_scoring.keys()) if self.custom_scoring is not None else \
                list(cfg.Cfg.scoring_funcs.regression_scoring_funcs_cv.keys())

            if self.trainer_configs.preprocess_strategy == 'pipeline':

                try:
                    cv_results = cross_validate(self.pipe_model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                                scoring=scoring_metrics)
                except Exception as e:
                    logging.error(f'Error during neural network cross-validation: {e}')
                    raise ValueError(
                        f'If using pipeline & cross validation check that you use the correct scoring parameters, '
                        f'you are using:{self.custom_scoring} . Error {e}')

                self.pipe_model.name = self.model.name
                logging.info("Neural network pipeline cross-validation completed successfully.")
                return cv_results

            elif self.trainer_configs.preprocess_strategy == 'custom':

                cv_results = cross_validate(self.model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                            scoring=scoring_metrics)
                logging.info("Neural network custom cross-validation completed successfully.")
                return cv_results

        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.trainer_configs.preprocess_strategy}")