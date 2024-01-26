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
    test_size = 0.20


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
