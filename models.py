import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from evaluators import Evaluator
from data_preprocessor import DataSpliter

import conf.config as cfg


# -----------------
# TEMPLATE PATTERN
# -----------------

class BaseModel:

    def execute_pipeline_steps(self, data, split_configs, pipe_steps=None):
        data = self.preprocess(data, pipe_steps)
        x_train, y_train, x_test, y_test = self.split(data, split_configs)

        model = self.train(x_train, y_train)

        return self.evaluate(model, x_test, y_test)

    def preprocess(self, data, pipe_steps):
        raise NotImplementedError

    @staticmethod
    def split(data, configs):
        s = DataSpliter(configs, data)
        return s.train_test_split()

    def train(self, x_train, y_train):
        raise NotImplementedError

    @staticmethod
    def evaluate(model, x_test, y_test):
        evaluator = Evaluator()
        return evaluator.evaluate(model, x_test, y_test)

    @staticmethod
    def setup_preprocessing_pipeline(model, pipe_steps):
        if pipe_steps is None:
            pipe_steps = []

        default_steps = [(model.name, model)]
        all_steps = pipe_steps + default_steps
        return Pipeline(all_steps)


# -----------------
# INDIVIDUAL MODELS
# -----------------


class LinearRegressorModel(BaseModel):
    def __init__(self, preprocess_strategy='pipeline'):
        self.pipe_model = None
        self.preprocess_strategy = preprocess_strategy
        self.model = LinearRegression()
        self.model.name = 'LinearRegression'

    @staticmethod
    def custom_preprocess(data):
        # Implement Random Forest-specific preprocessing
        # Example: Feature selection, custom transformations, etc.
        return data

    def preprocess(self, data, pipe_steps):
        if self.preprocess_strategy == 'pipeline':
            if self.preprocess_strategy == 'pipeline' and pipe_steps is None:
                logging.warning(f"Pipeline steps were not defined and preprocess_strategy is set to pipeline")

            self.pipe_model = self.setup_preprocessing_pipeline(model=self.model, pipe_steps=pipe_steps)

            return data
        elif self.preprocess_strategy == 'custom':
            return self.custom_preprocess(data)
        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.preprocess_strategy}")

    def train(self, x_train, y_train):
        if self.preprocess_strategy == 'pipeline':
            self.pipe_model.fit(x_train, y_train)
            self.pipe_model.name = self.model.name
            return self.pipe_model

        elif self.preprocess_strategy == 'custom':
            self.model.fit(x_train, y_train)
            return self.model
        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.preprocess_strategy}")


class RandomForestModel(BaseModel):
    def __init__(self, preprocess_strategy='pipeline'):
        self.pipe_model = None
        self.preprocess_strategy = preprocess_strategy
        model_configs = cfg.config_manager.get_config(model_name='RandomForestRegressor')
        self.model = RandomForestRegressor().set_params(**model_configs)
        self.model.name = 'RandomForestRegressor'

    @staticmethod
    def custom_preprocess(data):
        # Implement Random Forest-specific preprocessing
        # Example: Feature selection, custom transformations, etc.
        return data

    def preprocess(self, data, pipe_steps):
        if self.preprocess_strategy == 'pipeline':
            if self.preprocess_strategy == 'pipeline' and pipe_steps is None:
                logging.warning(f"Pipeline steps were not defined and preprocess_strategy is set to pipeline")

            self.pipe_model = self.setup_preprocessing_pipeline(model=self.model, pipe_steps=pipe_steps)

            return data
        elif self.preprocess_strategy == 'custom':
            return self.custom_preprocess(data)
        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.preprocess_strategy}")

    def train(self, x_train, y_train):
        if self.preprocess_strategy == 'pipeline':
            self.pipe_model.fit(x_train, y_train)
            self.pipe_model.name = self.model.name
            return self.pipe_model

        elif self.preprocess_strategy == 'custom':
            self.model.fit(x_train, y_train)
            return self.model
        else:
            raise ValueError(f"Unsupported preprocess strategy: {self.preprocess_strategy}")


# -----------------
# MODEL FACTORY
# -----------------
class ModelFactory:
    @staticmethod
    def create_regressor_model(model_type, preprocess_strategy='pipeline'):
        if model_type == 'linear_regression':
            return LinearRegressorModel(preprocess_strategy=preprocess_strategy)
        elif model_type == 'random_forest':
            return RandomForestModel(preprocess_strategy=preprocess_strategy)
        else:
            raise ValueError(f'Model type {model_type} not recognized.')

    def create_mlp_model(self):
        pass

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
