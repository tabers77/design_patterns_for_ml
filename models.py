import logging
from typing import Optional, List, Dict, Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from data_preprocessor import DataSpliter

from evaluators import Evaluator

import conf.config as cfg

from model_builders import MlpModelBuilder
from trainer import Trainer

from keras.models import Sequential


# -----------------
# TEMPLATE PATTERN
# -----------------
class BaseModel:
    """
    Base class for machine learning models using a template pattern.
    """

    def execute_pipeline_steps(self, data: pd.DataFrame,
                               split_configs: cfg.SplitConfigs,
                               trainer_configs: cfg.TrainerConfigs,
                               pipe_steps: Optional[List[Any]] = None) -> Any:
        """
        Execute the pipeline steps including preprocessing, training, and evaluation.

        Parameters:
        - data: Input data for the model.
        - data_spliter: Object responsible for splitting data.
        - trainer_configs: Configuration for the trainer.
        - pipe_steps: Optional list of pipeline steps.

        Returns:
        - Any: Result of the evaluation.
        """
        data = self.preprocess(data, pipe_steps)

        splits = self.split(data, split_configs)

        output = self.train(splits, split_configs, trainer_configs)

        return self.evaluate(model=output,
                             splits=splits,
                             split_configs=split_configs,
                             custom_scoring=trainer_configs.custom_scoring)

    def preprocess(self, data: Any, pipe_steps: Optional[List[Any]] = None) -> Any:
        raise NotImplementedError

    @staticmethod
    def split(data: Any, configs: Any) -> Any:
        """
        Split the data.

        Parameters:
        - data: Input data for the model.
        - configs: Configuration for data splitting.

        Returns:
        - Any: Split data.
        """
        s = DataSpliter(configs, data)
        return s.execute_split_steps()

    def train(self, splits: Any, split_configs: Any, trainer_configs: Any) -> Any:
        raise NotImplementedError

    def train(self, splits: Any, split_configs: Any, trainer_configs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def evaluate(model: Any, splits: Any, split_configs: Any, custom_scoring: Any) -> Any:
        """
        Evaluate the model.

        Parameters:
        - model: Trained model.
        - splits: Split data.
        - split_configs: Configuration for data splitting.
        - custom_scoring: Custom scoring configuration.

        Returns:
        - Any: Model evaluation output.
        """
        evaluator = Evaluator(split_configs=split_configs, custom_scoring=custom_scoring)
        return evaluator.evaluate(model, splits)

    @staticmethod
    def setup_preprocessing_pipeline(model: Any, pipe_steps: Optional[List[Any]]) -> Pipeline:
        """
        Setup preprocessing pipeline.

        Parameters:
        - model: Machine learning model.
        - pipe_steps: Optional list of pipeline steps.

        Returns:
        - Pipeline: Preprocessing pipeline.
        """
        if pipe_steps is None:
            pipe_steps = []

        default_steps = [(model.name, model)]
        all_steps = pipe_steps + default_steps
        return Pipeline(all_steps)

    @staticmethod
    def sanity_checks(data: Any, preprocess_strategy: str) -> None:
        """
        Perform sanity checks on the data.

        Parameters:
        - data: Input data for the model.
        - preprocess_strategy: Preprocessing strategy.

        Raises:
        - ValueError: If sanity checks fail.
        """
        if data.isnull().any().any() and preprocess_strategy == 'custom':
            raise ValueError('Data cant contain missing values with custom preprocess_strategy')


# -----------------
# INDIVIDUAL MODELS
# -----------------

class LinearRegressorModel(BaseModel):
    """
    Linear Regression model class.
    """

    def __init__(self, trainer_configs: Any):
        self.pipe_model: Optional[Pipeline] = None
        self.preprocess_strategy: str = trainer_configs.preprocess_strategy
        self.model: LinearRegression = LinearRegression()
        self.model.name: str = 'LinearRegression'

    @staticmethod
    def custom_preprocess(data: Any) -> Any:
        """
        Custom preprocessing for Linear Regression model.

        Parameters:
        - data: Input data.

        Returns:
        - Any: Preprocessed data.
        """
        # Implement LinearRegressorModel-specific preprocessing
        # Example: Feature selection, custom transformations, etc.
        return data

    def preprocess(self, data: Any, pipe_steps: Optional[List[Any]] = None) -> Any:
        """
        Preprocess the input data.

        Parameters:
        - data: Input data for the model.
        - pipe_steps: Optional list of pipeline steps.

        Returns:
        - Any: Preprocessed data.
        """
        # Sanity checks
        self.sanity_checks(data, self.preprocess_strategy)

        if self.preprocess_strategy == 'pipeline':
            if self.preprocess_strategy == 'pipeline' and pipe_steps is None:
                logging.warning("Pipeline steps were not defined and preprocess_strategy is set to pipeline")

            self.pipe_model = self.setup_preprocessing_pipeline(model=self.model, pipe_steps=pipe_steps)

            return data
        elif self.preprocess_strategy == 'custom':
            return self.custom_preprocess(data)
        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.preprocess_strategy}")

    def train(self, splits: Any, split_configs: Any, trainer_configs: Any) -> Any:
        """
        Train the Linear Regression model.

        Parameters:
        - splits: Split data.
        - split_configs: Configuration for data splitting.
        - trainer_configs: Configuration for the trainer.

        Returns:
        - Any: Model training output.
        """
        trainer = Trainer(splits=splits,
                          split_configs=split_configs,
                          trainer_configs=trainer_configs,
                          model=self.model,
                          pipe_model=self.pipe_model)
        return trainer.base_train()


class RandomForestModel(BaseModel):
    """
    Random Forest Regressor model class.
    """

    def __init__(self, trainer_configs: Any):
        self.pipe_model: Optional[Pipeline] = None
        self.preprocess_strategy: str = trainer_configs.preprocess_strategy
        model_configs: Dict[str, Any] = cfg.config_manager.get_config(model_name='RandomForestRegressor')
        self.model: RandomForestRegressor = RandomForestRegressor().set_params(**model_configs)
        self.model.name: str = 'RandomForestRegressor'

    @staticmethod
    def custom_preprocess(data: Any) -> Any:
        """
        Custom preprocessing for Random Forest model.

        Parameters:
        - data: Input data.

        Returns:
        - Any: Preprocessed data.
        """
        # Implement Random Forest-specific preprocessing
        # Example: Feature selection, custom transformations, etc.
        return data

    def preprocess(self, data: Any, pipe_steps: Optional[List[Any]] = None) -> Any:
        """
        Preprocess the input data.

        Parameters:
        - data: Input data for the model.
        - pipe_steps: Optional list of pipeline steps.

        Returns:
        - Any: Preprocessed data.
        """
        # Sanity checks
        self.sanity_checks(data, self.preprocess_strategy)

        if self.preprocess_strategy == 'pipeline':
            if self.preprocess_strategy == 'pipeline' and pipe_steps is None:
                logging.warning("Pipeline steps were not defined and preprocess_strategy is set to pipeline")

            self.pipe_model = self.setup_preprocessing_pipeline(model=self.model, pipe_steps=pipe_steps)

            return data
        elif self.preprocess_strategy == 'custom':
            return self.custom_preprocess(data)
        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.preprocess_strategy}")

    def train(self, splits: Any, split_configs: Any, trainer_configs: Any) -> Any:
        """
        Train the Random Forest model.

        Parameters:
        - splits: Split data.
        - split_configs: Configuration for data splitting.
        - trainer_configs: Configuration for the trainer.

        Returns:
        - Any: Model training output.
        """
        trainer = Trainer(splits=splits,
                          split_configs=split_configs,
                          trainer_configs=trainer_configs,
                          model=self.model,
                          pipe_model=self.pipe_model)
        return trainer.base_train()


class MlpModel(BaseModel):
    """
    Multi-Layer Perceptron (MLP) model class.
    """

    def __init__(self, trainer_configs: Any):
        self.pipe_model: Optional[Pipeline] = None
        self.preprocess_strategy: str = trainer_configs.preprocess_strategy
        self.model: Sequential = MlpModelBuilder(trainer_configs).build_mlp()

    @staticmethod
    def custom_preprocess(data: Any) -> Any:
        """
        Custom preprocessing for MLP model.

        Parameters:
        - data: Input data.

        Returns:
        - Any: Preprocessed data.
        """
        # Implement MLP-specific preprocessing
        # Example: Feature selection, custom transformations, etc.
        return data

    def preprocess(self, data: Any, pipe_steps: Optional[List[Any]] = None) -> Any:
        """
        Preprocess the input data.

        Parameters:
        - data: Input data for the model.
        - pipe_steps: Optional list of pipeline steps.

        Returns:
        - Any: Preprocessed data.
        """
        # Sanity checks
        self.sanity_checks(data, self.preprocess_strategy)

        if self.preprocess_strategy == 'pipeline':
            if self.preprocess_strategy == 'pipeline' and pipe_steps is None:
                logging.warning("Pipeline steps were not defined and preprocess_strategy is set to pipeline")

            self.pipe_model = self.setup_preprocessing_pipeline(model=self.model, pipe_steps=pipe_steps)

            return data
        elif self.preprocess_strategy == 'custom':
            return self.custom_preprocess(data)
        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.preprocess_strategy}")

    def train(self, splits: Any, split_configs: Any, trainer_configs: Any) -> Any:
        """
        Train the MLP model.

        Parameters:
        - splits: Split data.
        - split_configs: Configuration for data splitting.
        - trainer_configs: Configuration for the trainer.

        Returns:
        - Any: Model training output.
        """
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
    """
    Factory class for creating regression models.
    """

    @staticmethod
    def create_regressor_model(model_type: str, trainer_configs: Any) -> BaseModel:
        """
        Create a regression model.

        Parameters:
        - model_type: Type of regression model to create.
        - trainer_configs: Configuration for the trainer.

        Returns:
        - BaseModel: Instance of the created regression model.
        """
        if model_type == 'linear_regression':
            return LinearRegressorModel(trainer_configs=trainer_configs)
        elif model_type == 'random_forest':
            return RandomForestModel(trainer_configs=trainer_configs)
        else:
            raise ValueError(f'Model type {model_type} not recognized.')

    @staticmethod
    def create_mlp_model(trainer_configs: Any) -> BaseModel:
        """
        Create an MLP model.

        Parameters:
        - trainer_configs: Configuration for the trainer.

        Returns:
        - BaseModel: Instance of the created MLP model.
        """
        return MlpModel(trainer_configs=trainer_configs)
