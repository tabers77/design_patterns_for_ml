# from semla.conf.config_constants import ConfigConstants
# from semla.conf.config_paths import ConfigPaths
# from semla.conf.config_model import ConfigModel
# from semla.conf.config_columns import ConfigColumns, ConfigColumnsLowerCase
# from semla.conf.config_features import ConfigFeatures
from dataclasses import dataclass
from dataclasses_json import dataclass_json


class GlobalConfigs:
    def __init__(self, local_run: bool = True, sample_size: float = 1, random_state: int = 42):
        self.local_run = local_run
        self.sample_size = sample_size
        self.random_state = random_state


class SplitConfigs:
    def __init__(self, target_col_name=None, train_size=0.20):
        self.target_col_name = target_col_name
        self.train_size = train_size


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
config_manager.set_config("RandomForest", n_estimators=100, max_depth=None)  # Adjust parameters as needed


@dataclass_json
@dataclass(frozen=True)
class Cfg:
    global_configs: GlobalConfigs = GlobalConfigs()
    split_configs: SplitConfigs = SplitConfigs()

    # constants: ConfigConstants = ConfigConstants()
    # paths: ConfigPaths = ConfigPaths()
    # model: ConfigModel = ConfigModel()
    # columns: ConfigColumns = ConfigColumns()
    # columns_lower_case: ConfigColumnsLowerCase = ConfigColumnsLowerCase()
    # features: ConfigFeatures = ConfigFeatures()
