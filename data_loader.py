import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from typing import Optional
from functools import lru_cache

MISSING_VALUE_RATIO = 0.1  # 10% of data

class DataLoader:
    @staticmethod
    @lru_cache(maxsize=10)
    def load_diabetes_data(with_missing_values: Optional[bool] = False) -> pd.DataFrame:
        """
        Load the Diabetes dataset with optional caching to improve load times.

        Parameters:
        - with_missing_values (bool): If True, introduce missing values to the features.

        Returns:
        pd.DataFrame: DataFrame containing the feature and target variables.
        """
        diabetes_data = load_diabetes()
        data = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
        data['target'] = diabetes_data.target

        if with_missing_values:
            features_columns = diabetes_data.feature_names
            num_samples = int(MISSING_VALUE_RATIO * len(data))
            idx = np.random.choice(data.index, num_samples, replace=False)
            data.loc[idx, features_columns] = np.nan

        return data

    @staticmethod
    def clear_cache():
        DataLoader.load_diabetes_data.cache_clear()