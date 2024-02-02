from sklearn.model_selection import cross_validate

from conf import config as cfg


class Trainer:
    def __init__(self, splits, split_configs, trainer_configs, model=None, pipe_model=None):
        self.splits = splits
        self.split_configs = split_configs
        self.trainer_configs = trainer_configs
        self.model = model
        self.pipe_model = pipe_model
        self.custom_scoring = self.trainer_configs.custom_scoring

    def base_train(self):

        if self.split_configs.split_policy == 'feature_target':
            if self.trainer_configs.preprocess_strategy == 'pipeline':
                self.pipe_model.fit(self.splits.x_train, self.splits.y_train)
                self.pipe_model.name = self.model.name
                return self.pipe_model

            elif self.trainer_configs.preprocess_strategy == 'custom':
                self.model.fit(self.splits.x_train, self.splits.y_train)
                return self.model

        elif self.split_configs.split_policy == 'x_y_splits_only':

            # Observe that model evaluation is already performed here:
            scoring_metrics = list(self.custom_scoring.keys()) if self.custom_scoring is not None else \
                list(cfg.Cfg.scoring_funcs.regression_scoring_funcs_cv.keys())

            if self.trainer_configs.preprocess_strategy == 'pipeline':

                # Perform cross-validation
                try:
                    cv_results = cross_validate(self.pipe_model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                                scoring=scoring_metrics)
                except Exception as e:
                    raise ValueError(
                        f'If using pipeline & cross validation check that you use the correct scoring parameters, '
                        f'you are using:{self.custom_scoring} . Error {e}')

                self.pipe_model.name = self.model.name
                return cv_results

            elif self.trainer_configs.preprocess_strategy == 'custom':

                # Perform cross-validation
                cv_results = cross_validate(self.model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                            scoring=scoring_metrics)

                return cv_results

        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.trainer_configs.preprocess_strategy}")

    def nn_train(self):

        """Trainer for neural networks"""

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
            scoring_metrics = list(self.custom_scoring.keys()) if self.custom_scoring is not None else \
                list(cfg.Cfg.scoring_funcs.regression_scoring_funcs_cv.keys())

            if self.trainer_configs.preprocess_strategy == 'pipeline':

                # Perform cross-validation
                try:
                    cv_results = cross_validate(self.pipe_model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                                scoring=scoring_metrics)
                except Exception as e:
                    raise ValueError(
                        f'If using pipeline & cross validation check that you use the correct scoring parameters, '
                        f'you are using:{self.custom_scoring} . Error {e}')

                self.pipe_model.name = self.model.name
                return cv_results

            elif self.trainer_configs.preprocess_strategy == 'custom':

                # Perform cross-validation
                cv_results = cross_validate(self.model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                            scoring=scoring_metrics)

                return cv_results

        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.trainer_configs.preprocess_strategy}")
