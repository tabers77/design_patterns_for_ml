from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from conf.constants import Constants
import sklearn.metrics as m
from typing import Any


@dataclass_json
@dataclass(frozen=True)
class GlobalConfigs:
    local_run: bool = True
    sample_size: float = 1
    random_state: int = 42


@dataclass_json
@dataclass(frozen=True)
class SplitConfigs:
    target_col_name: str = None
    train_size: float = 0.20
    cv: int = 5
    split_policy: str = 'feature_target'


@dataclass_json
@dataclass(frozen=True)
class TrainerConfigs:
    preprocess_strategy: str = 'custom'
    scorer: Any = None
    input_dim: int = None


@dataclass_json
@dataclass(frozen=True)
class ModelConfig:
    _config: dict = field(default_factory=dict)

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
    constants: Constants = Constants()
    scoring_funcs: ScoringFuncs = ScoringFuncs()
