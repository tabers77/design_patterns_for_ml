from sklearn.model_selection import cross_validate
from conf import config as cfg
from typing import Any, Optional, Union, Dict


class Trainer:
    """
    Trainer class for handling training and cross-validation of models.
    """

    def __init__(self, splits: Any,  trainer_configs: Any, model: Optional[Any] = None,
                 pipe_model: Optional[Any] = None) -> None:
        self.splits = splits
        self.split_configs = trainer_configs.scorer.split_configs
        self.trainer_configs = trainer_configs
        self.model = model
        self.pipe_model = pipe_model
        # self.custom_scoring = self.trainer_configs.custom_scoring

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
                return self.pipe_model

            elif self.trainer_configs.preprocess_strategy == 'custom':

                self.model.fit(self.splits.x_train, self.splits.y_train)
                return self.model

        elif self.split_configs.split_policy == 'x_y_splits_only':
            # Observe that cross_validate requires a list
            scoring_metrics = list(self.trainer_configs.scorer.get_scoring_funcs().keys())

            if self.trainer_configs.preprocess_strategy == 'pipeline':

                # Perform cross-validation
                try:
                    cv_results = cross_validate(self.pipe_model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                                scoring=scoring_metrics)
                except Exception as e:
                    raise ValueError(
                        f'If using pipeline & cross validation check that you use the correct scoring parameters, '
                        f'you are using:{self.trainer_configs.scorer.custom_scoring} . Error {e}')

                self.pipe_model.name = self.model.name

                cv_results['name'] = self.model.name

                return cv_results

            elif self.trainer_configs.preprocess_strategy == 'custom':

                # Perform cross-validation
                cv_results = cross_validate(self.model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                            scoring=scoring_metrics)

                cv_results['name'] = self.model.name

                return cv_results

        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.split_configs.split_policy}")

    def nn_train(self) -> Union[Any, Dict[str, Any]]:
        """
        Perform training for neural networks.

        Returns:
        - Union[Any, Dict[str, Any]]: Trained model or cross-validation results.
        """

        if self.split_configs.split_policy == 'feature_target':
            if self.trainer_configs.preprocess_strategy == 'pipeline':
                raise NotImplementedError
                # self.pipe_model.fit(self.splits.x_train, self.splits.y_train)
                # self.pipe_model.name = self.model.name
                # return self.pipe_model

            elif self.trainer_configs.preprocess_strategy == 'custom':
                self.model.fit(self.splits.x_train,
                               self.splits.y_train,
                               epochs=50,
                               batch_size=32,
                               validation_data=(self.splits.x_test, self.splits.y_test), verbose=2)
                return self.model

        elif self.split_configs.split_policy == 'x_y_splits_only':

            # Observe that model evaluation is already performed here:
            scoring_metrics = list(self.trainer_configs.scorer.custom_scoring.keys()) if \
                self.trainer_configs.scorer.custom_scoring is not None else \
                list(cfg.Cfg.scoring_funcs.regression_scoring_funcs_cv.keys())

            if self.trainer_configs.preprocess_strategy == 'pipeline':

                # Perform cross-validation
                try:
                    cv_results = cross_validate(self.pipe_model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                                scoring=scoring_metrics)
                except Exception as e:
                    raise ValueError(
                        f'If using pipeline & cross validation check that you use the correct scoring parameters, '
                        f'you are using:{self.trainer_configs.scorer.custom_scoring} . Error {e}')

                # TODO: IMPLEMENT: cv_results['name'] = self.model.name

                self.pipe_model.name = self.model.name
                return cv_results

            elif self.trainer_configs.preprocess_strategy == 'custom':

                # Perform cross-validation
                cv_results = cross_validate(self.model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                            scoring=scoring_metrics)
                # TODO: IMPLEMENT: cv_results['name'] = self.model.name

                return cv_results

        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.trainer_configs.preprocess_strategy}")
