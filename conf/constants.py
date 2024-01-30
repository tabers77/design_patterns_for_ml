from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class Constants:
    fixed_columns = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6',
                     'target']
