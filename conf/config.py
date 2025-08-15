from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from conf.constants import Constants
import sklearn.metrics as m
from typing import Optional, Dict

@dataclass_json
@dataclass
class GlobalConfigs:
    local_run: bool = True
    sample_size: float = 1.0
    random_state: int = 42

@dataclass_json
@dataclass
class SplitConfigs:
    target_col_name: Optional[str] = None
    train_size: float = 0.20
    cv: int = 5
    split_policy: str = 'feature_target'

@dataclass_json
@dataclass
class TrainerConfigs:
    preprocess_strategy: str = 'custom'
    custom_scoring: Optional[Dict[str, float]] = None
    input_dim: Optional[int] = None

@dataclass_json
@dataclass
class ModelConfig:
    _config: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def set_config(self, model_name: str, **kwargs: float) -> None:
        self._config[model_name] = kwargs

    def get_config(self, model_name: str) -> Dict[str, float]:
        return self._config.get(model_name, {})

@dataclass_json
@dataclass
class ScoringFuncs:
    regression_scoring_funcs: Dict[str, callable] = {
        'mean_squared_error': m.mean_squared_error,
        'mean_absolute_error': m.mean_absolute_error,
        'mean_absolute_percentage_error': m.mean_absolute_percentage_error,
    }
    regression_scoring_funcs_cv: Dict[str, callable] = {
        'neg_' + k: v for k, v in regression_scoring_funcs.items()
    }

@dataclass_json
@dataclass
class Cfg:
    constants: Constants = Constants()
    scoring_funcs: ScoringFuncs = ScoringFuncs()