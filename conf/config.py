from dataclasses import dataclass
from dataclasses_json import dataclass_json
from conf.constants import Constants
import sklearn.metrics as m


class GlobalConfigs:
    def __init__(self, local_run: bool = True, sample_size: float = 1, random_state: int = 42):
        self.local_run = local_run
        self.sample_size = sample_size
        self.random_state = random_state


class SplitConfigs:
    def __init__(self, target_col_name=None, train_size=0.20, cv=5, split_policy='feature_target'):
        self.target_col_name = target_col_name
        self.train_size = train_size
        self.cv = cv
        self.split_policy = split_policy


class TrainerConfigs:
    def __init__(self, preprocess_strategy='custom', custom_scoring: dict = None, input_dim=None):
        self.preprocess_strategy = preprocess_strategy
        self.custom_scoring = custom_scoring
        self.input_dim = input_dim


# Singleton for Model Configuration Management
class ModelConfig:
    _instance = None
    _config = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelConfig, cls).__new__(cls)
        return cls._instance

    def set_config(self, model_name, **kwargs):
        self._config[model_name] = kwargs

    def get_config(self, model_name):
        return self._config.get(model_name, {})


# Usage
config_manager = ModelConfig()
config_manager.set_config("RandomForestRegressor",
                          n_estimators=100,
                          max_depth=None,
                          random_state=42)  # Adjust parameters as needed


@dataclass_json
@dataclass(frozen=True)
class ScoringFuncs:
    regression_scoring_funcs = {'mean_squared_error': m.mean_squared_error,
                                'mean_absolute_error': m.mean_absolute_error,
                                'mean_absolute_percentage_error': m.mean_absolute_percentage_error,
                                }
    regression_scoring_funcs_cv = {'neg_' + k: v for k, v in regression_scoring_funcs.items()}


@dataclass_json
@dataclass(frozen=True)
class Cfg:
    global_configs: GlobalConfigs = GlobalConfigs()
    constants: Constants = Constants()
    scoring_funcs: ScoringFuncs = ScoringFuncs()
