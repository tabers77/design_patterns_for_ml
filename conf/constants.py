from dataclasses import dataclass
from dataclasses_json import dataclass_json
from matplotlib.colors import ListedColormap


@dataclass_json
@dataclass(frozen=True)
class Constants:
    fixed_columns = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6',
                     'target']


color_list = ["#A5D7E8", "#576CBC"]  # "#19376D", "#0B2447"
cmap_custom = ListedColormap(color_list)
