import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes


class DataLoder:
    @staticmethod
    def load_diabetes_data(with_missing_values=False):
        # Load Boston Housing dataset
        diabetes_df = load_diabetes()
        # Create a DataFrame with the feature and target variables
        data = pd.DataFrame(diabetes_df.data, columns=diabetes_df.feature_names)
        data['target'] = diabetes_df.target

        if with_missing_values:
            # Introduce missing values only to the features
            features_columns = diabetes_df.feature_names
            idx = data[features_columns].sample(n=20).index
            data.loc[idx, features_columns] = np.nan

        return data
