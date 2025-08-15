import logging
from typing import Optional, List, Dict, Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from evaluators import Evaluator
from data_preprocessor import DataSpliter

import conf.config as cfg

from trainer import Trainer


class BaseModel:
    def execute_pipeline_steps(self, data: pd.DataFrame, split_configs: cfg.SplitConfigs,
                               trainer_configs: cfg.TrainerConfigs,
                               pipe_steps: Optional[List[Any]] = None) -> Any:
        data = self.preprocess(data, pipe_steps)
        splits = self.split(data, split_configs)
        output = self.train(splits, split_configs, trainer_configs)
        return self.evaluate(model=output, splits=splits, split_configs=split_configs, custom_scoring=trainer_configs.custom_scoring)

    def preprocess(self, data: Any, pipe_steps: Optional[List[Any]] = None) -> Any:
        raise NotImplementedError

    @staticmethod
    def split(data: Any, configs: Any) -> Any:
        s = DataSpliter(configs, data)
        return s.execute_split_steps()

    def train(self, splits: Any, split_configs: Any, trainer_configs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def evaluate(model: Any, splits: Any, split_configs: Any, custom_scoring: Any) -> Any:
        evaluator = Evaluator(split_configs=split_configs, custom_scoring=custom_scoring)
        return evaluator.evaluate(model, splits)

    @staticmethod
    def setup_preprocessing_pipeline(model: Any, pipe_steps: Optional[List[Any]]) -> Pipeline:
        if pipe_steps is None:
            pipe_steps = []
        default_steps = [(model.name, model)]
        all_steps = pipe_steps + default_steps
        return Pipeline(all_steps)

    @staticmethod
    def sanity_checks(data: Any, preprocess_strategy: str) -> None:
        if data.isnull().any().any() and preprocess_strategy == 'custom':
            raise ValueError('Data cannot contain missing values with custom preprocess_strategy')


class LinearRegressorModel(BaseModel):
    def __init__(self, trainer_configs: cfg.TrainerConfigs):
        self.pipe_model: Optional[Pipeline] = None
        self.preprocess_strategy: str = trainer_configs.preprocess_strategy
        self.model: LinearRegression = LinearRegression()
        self.model.name: str = 'LinearRegression'

    @staticmethod
    def custom_preprocess(data: Any) -> Any:
        return data

    def preprocess(self, data: Any, pipe_steps: Optional[List[Any]] = None) -> Any:
        self.sanity_checks(data, self.preprocess_strategy)
        if self.preprocess_strategy == 'pipeline':
            if pipe_steps is None:
                logging.warning("Pipeline steps were not defined and preprocess_strategy is set to pipeline")
            self.pipe_model = self.setup_preprocessing_pipeline(model=self.model, pipe_steps=pipe_steps)
            return data
        elif self.preprocess_strategy == 'custom':
            return self.custom_preprocess(data)
        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.preprocess_strategy}")

    def train(self, splits: Any, split_configs: Any, trainer_configs: Any) -> Any:
        trainer = Trainer(splits=splits, split_configs=split_configs, trainer_configs=trainer_configs, model=self.model, pipe_model=self.pipe_model)
        return trainer.base_train()


class RandomForestModel(BaseModel):
    def __init__(self, trainer_configs: cfg.TrainerConfigs):
        self.pipe_model: Optional[Pipeline] = None
        self.preprocess_strategy: str = trainer_configs.preprocess_strategy
        model_configs: Dict[str, Any] = cfg.config_manager.get_config(model_name='RandomForestRegressor')
        self.model: RandomForestRegressor = RandomForestRegressor(**model_configs)
        self.model.name: str = 'RandomForestRegressor'

    @staticmethod
    def custom_preprocess(data: Any) -> Any:
        return data

    def preprocess(self, data: Any, pipe_steps: Optional[List[Any]] = None) -> Any:
        self.sanity_checks(data, self.preprocess_strategy)
        if self.preprocess_strategy == 'pipeline':
            if pipe_steps is None:
                logging.warning("Pipeline steps were not defined and preprocess_strategy is set to pipeline")
            self.pipe_model = self.setup_preprocessing_pipeline(model=self.model, pipe_steps=pipe_steps)
            return data
        elif self.preprocess_strategy == 'custom':
            return self.custom_preprocess(data)