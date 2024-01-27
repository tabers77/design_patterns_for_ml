from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from data_preprocessor import DataSpliter

import conf.config as cfg


# TEMPLATE PATTERN
class BaseModel:

    def execute_pipeline_steps(self, data, configs):
        data = self.preprocess(data)
        x_train, y_train, x_test, y_test = self.split(data, configs)

        model = self.train(x_train, y_train)
        self.evaluate(model, x_test, y_test)

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
        predictions = model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model mean_squared_error: {mse:.2f}")


class SVMModel(BaseModel):
    def preprocess(self, data):
        # Implement SVM-specific preprocessing
        pass

    def train(self, data, labels):
        # Implement SVM training
        pass


class RandomForestModel(BaseModel):
    def __init__(self):
        model_configs = cfg.config_manager.get_config(model_name='RandomForest')
        self.model = RandomForestRegressor().set_params(**model_configs)

    def preprocess(self, data):
        # Implement Random Forest-specific preprocessing
        return data

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return self.model
        # Implement Random Forest training


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
# CREATE A BUILDER FOR MLP
# class RandomForestModelBuilder:
#     def __init__(self):
#         self.model = RandomForestModel()
#
#     def set_params(self, params):
#         self.model.set_params(params)
#         return self
#
#     def build(self):
#         return self.model

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
