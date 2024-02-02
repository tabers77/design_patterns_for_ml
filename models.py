import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from evaluators import Evaluator
from data_preprocessor import DataSpliter

import conf.config as cfg

from model_builders import MlpModelBuilder
from trainer import Trainer


# -----------------
# TEMPLATE PATTERN
# -----------------
class BaseModel:

    def execute_pipeline_steps(self, data, split_configs, trainer_configs, pipe_steps=None):
        data = self.preprocess(data, pipe_steps)

        splits = self.split(data, split_configs)

        output = self.train(splits, split_configs, trainer_configs)

        return self.evaluate(model=output,
                             splits=splits,
                             split_configs=split_configs,
                             custom_scoring=trainer_configs.custom_scoring)

    def preprocess(self, data, pipe_steps=None):
        raise NotImplementedError

    @staticmethod
    def split(data, configs):
        s = DataSpliter(configs, data)
        return s.execute_split_steps()

    def train(self, splits, split_configs, trainer_configs):
        raise NotImplementedError

    @staticmethod
    def evaluate(model, splits, split_configs, custom_scoring):
        evaluator = Evaluator(split_configs=split_configs, custom_scoring=custom_scoring)
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

    def preprocess(self, data, pipe_steps=None):
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

    def preprocess(self, data, pipe_steps=None):
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


class MlpModel(BaseModel):
    def __init__(self, trainer_configs):
        self.pipe_model = None
        self.preprocess_strategy = trainer_configs.preprocess_strategy
        # model_configs = cfg.config_manager.get_config(model_name='RandomForestRegressor')
        self.model = MlpModelBuilder(trainer_configs).build_mlp()
        # self.model.name = 'MlP model'

    @staticmethod
    def custom_preprocess(data):
        # Implement Random Forest-specific preprocessing
        # Example: Feature selection, custom transformations, etc.
        return data

    def preprocess(self, data, pipe_steps=None):
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
        return trainer.nn_train()


# --------
# FACTORY
# -------

class ModelFactory:
    @staticmethod
    def create_regressor_model(model_type, trainer_configs):
        if model_type == 'linear_regression':
            return LinearRegressorModel(trainer_configs=trainer_configs)
        elif model_type == 'random_forest':
            return RandomForestModel(trainer_configs=trainer_configs)
        else:
            raise ValueError(f'Model type {model_type} not recognized.')

    @staticmethod
    def create_mlp_model(trainer_configs):
        return MlpModel(trainer_configs=trainer_configs)
