from sklearn.ensemble import RandomForestRegressor

from evaluators import Evaluator
from data_preprocessor import DataSpliter

import conf.config as cfg


# -----------------
# TEMPLATE PATTERN
# -----------------

class BaseModel:

    def execute_pipeline_steps(self, data, configs):
        data = self.preprocess(data)
        x_train, y_train, x_test, y_test = self.split(data, configs)

        model = self.train(x_train, y_train)

        return self.evaluate(model, x_test, y_test)

    def preprocess(self, data):
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


# -----------------
# INDIVIDUAL MODELS
# -----------------
class SVMModel(BaseModel):
    def preprocess(self, data):
        # Implement SVM-specific preprocessing
        pass

    def train(self, data, labels):
        # Implement SVM training
        pass


class RandomForestModel(BaseModel):
    def __init__(self):
        model_configs = cfg.config_manager.get_config(model_name='RandomForestRegressor')
        self.model = RandomForestRegressor().set_params(**model_configs)
        self.model.name = 'RandomForestRegressor'

    def preprocess(self, data):
        # Implement Random Forest-specific preprocessing
        return data

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return self.model


# -----------------
# MODEL FACTORY
# -----------------
class ModelFactory:
    @staticmethod
    def create_regressor_model(model_type):
        if model_type == 'custom_model':
            pass
        elif model_type == 'random_forest':
            return RandomForestModel()
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


# USE THIS
# X, y = make_classification(random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create and train pipeline
# pipe = Pipeline([('imputer', SimpleImputer(strategy="median")),
#                  ('scaler', StandardScaler()),
#                  ('svc', SVC())])
# pipe.fit(X_train, y_train)

# Evaluate the pipeline
# >>> pipe.score(X_test, y_test)
# 0.88
