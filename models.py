import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from evaluators import Evaluator
from data_preprocessor import DataSpliter

import conf.config as cfg


# -----------------
# TEMPLATE PATTERN
# -----------------
class BaseModel:

    def execute_pipeline_steps(self, data, split_configs, trainer_configs, pipe_steps=None):
        data = self.preprocess(data, pipe_steps)

        splits = self.split(data, split_configs)

        output = self.train(splits, split_configs, trainer_configs)

        return self.evaluate(output, splits, split_configs)

    def preprocess(self, data, pipe_steps):
        raise NotImplementedError

    @staticmethod
    def split(data, configs):
        s = DataSpliter(configs, data)
        return s.execute_split_steps()

    def train(self, splits, split_configs, trainer_configs):
        raise NotImplementedError

    @staticmethod
    def evaluate(model, splits, split_configs):
        evaluator = Evaluator(split_configs=split_configs)
        return evaluator.evaluate(model, splits)

    @staticmethod
    def setup_preprocessing_pipeline(model, pipe_steps):
        if pipe_steps is None:
            pipe_steps = []

        default_steps = [(model.name, model)]
        all_steps = pipe_steps + default_steps
        return Pipeline(all_steps)

    @staticmethod
    def sanity_checks(data, preprocess_strategy):
        if data.isnull().any().any() and preprocess_strategy == 'custom':
            raise ValueError('Data cant contain missing values with custom preprocess_strategy')


# -----------------
# INDIVIDUAL MODELS
# -----------------

# TODO: FIX LINEAR REGRESSION

class LinearRegressorModel(BaseModel):
    def __init__(self, trainer_configs):
        self.pipe_model = None
        self.preprocess_strategy = trainer_configs.preprocess_strategy
        self.model = LinearRegression()
        self.model.name = 'LinearRegression'

    @staticmethod
    def custom_preprocess(data):
        # Implement LinearRegressorModel-specific preprocessing
        # Example: Feature selection, custom transformations, etc.
        return data

    def preprocess(self, data, pipe_steps):
        # Sanity checks
        self.sanity_checks(data, self.preprocess_strategy)

        if self.preprocess_strategy == 'pipeline':
            if self.preprocess_strategy == 'pipeline' and pipe_steps is None:
                logging.warning(f"Pipeline steps were not defined and preprocess_strategy is set to pipeline")

            self.pipe_model = self.setup_preprocessing_pipeline(model=self.model, pipe_steps=pipe_steps)

            return data
        elif self.preprocess_strategy == 'custom':
            return self.custom_preprocess(data)
        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.preprocess_strategy}")

    def train(self, splits, split_configs, trainer_configs):
        trainer = Trainer(splits=splits,
                          split_configs=split_configs,
                          trainer_configs=trainer_configs,
                          model=self.model,
                          pipe_model=self.pipe_model)
        return trainer.base_train()


class RandomForestModel(BaseModel):
    def __init__(self, trainer_configs):
        self.pipe_model = None
        self.preprocess_strategy = trainer_configs.preprocess_strategy
        model_configs = cfg.config_manager.get_config(model_name='RandomForestRegressor')
        self.model = RandomForestRegressor().set_params(**model_configs)
        self.model.name = 'RandomForestRegressor'

    @staticmethod
    def custom_preprocess(data):
        # Implement Random Forest-specific preprocessing
        # Example: Feature selection, custom transformations, etc.
        return data

    def preprocess(self, data, pipe_steps):
        # Sanity checks
        self.sanity_checks(data, self.preprocess_strategy)

        if self.preprocess_strategy == 'pipeline':
            if self.preprocess_strategy == 'pipeline' and pipe_steps is None:
                logging.warning(f"Pipeline steps were not defined and preprocess_strategy is set to pipeline")

            self.pipe_model = self.setup_preprocessing_pipeline(model=self.model, pipe_steps=pipe_steps)

            return data
        elif self.preprocess_strategy == 'custom':
            return self.custom_preprocess(data)
        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.preprocess_strategy}")

    def train(self, splits, split_configs, trainer_configs):
        trainer = Trainer(splits=splits,
                          split_configs=split_configs,
                          trainer_configs=trainer_configs,
                          model=self.model,
                          pipe_model=self.pipe_model)
        return trainer.base_train()


# -----------------
# MODEL FACTORY
# -----------------
class ModelFactory:
    @staticmethod
    def create_regressor_model(model_type, trainer_configs):
        if model_type == 'linear_regression':
            return LinearRegressorModel(trainer_configs=trainer_configs)
        elif model_type == 'random_forest':
            return RandomForestModel(trainer_configs=trainer_configs)
        else:
            raise ValueError(f'Model type {model_type} not recognized.')

    def create_mlp_model(self):
        pass


class Trainer:
    def __init__(self, splits, split_configs, trainer_configs, model=None, pipe_model=None):
        self.splits = splits
        self.split_configs = split_configs
        self.trainer_configs = trainer_configs
        self.model = model
        self.pipe_model = pipe_model
        # self.custom_scoring = self.trainer_configs.custom_scoring  # TODO: IMPLEMENT

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
            scoring_metrics = list(cfg.Cfg.scoring_funcs.regression_scoring_funcs_cv.keys())

            if self.trainer_configs.preprocess_strategy == 'pipeline':

                # Perform cross-validation
                cv_results = cross_validate(self.pipe_model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                            scoring=scoring_metrics)

                self.pipe_model.name = self.model.name
                return cv_results

            elif self.trainer_configs.preprocess_strategy == 'custom':

                # Perform cross-validation
                cv_results = cross_validate(self.model, self.splits.x, self.splits.y, cv=self.split_configs.cv,
                                            scoring=scoring_metrics)

                return cv_results

        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.trainer_configs.preprocess_strategy}")

# ------------
# NEXT STEPS :
# ------------

# # BUILDER PATTERN
# class MlpModelBuilder:
#     def __init__(self):
#         self.model = RandomForestModel()
#
#     # def set_params(self, params):
#     #     self.model.set_params(params)
#     #     return self
#
#     def build_mlp(self):
#         """
#           Info: MLP for baseline model creation. Regulator helps you to increase or decrease n_neurons
#
#           Args:
#               x:
#               y:
#               activation_f_type:
#               optimizer:
#               regulator:
#               hl_activation:
#               evaluation_metric:
#
#           Returns:
#
#           """
#         regulator= 20
#         n_inputs = x.shape[1]
#         n_outputs = int(y.nunique())
#
#         #o_activation, loss = get_mlp_initial_params(activation_f_type=activation_f_type)
#
#         n_neurons = int(np.sqrt(n_inputs * n_outputs) * regulator)
#
#         print(f'Number of neurons: {n_neurons}-{int(n_neurons / 2.5)}-{int(n_neurons / 5.5)}')
#
#         model = Sequential()
#
#         model.add(Dense(n_neurons, input_dim=n_inputs, activation=hl_activation))
#         model.add(Dropout(0.3))
#
#         model.add(Dense(int(n_neurons / 2.5), activation=hl_activation))
#         model.add(Dropout(0.3))
#
#         model.add(Dense(int(n_neurons / 5.5), activation=hl_activation))
#         model.add(Dropout(0.1))
#
#         model.add(Dense(n_outputs, activation=o_activation))  # output layer
#
#         model.compile(loss=loss, optimizer=optimizer, metrics=[evaluation_metric])
#
#         return model
#
#     def set_params(self):
#         pass
