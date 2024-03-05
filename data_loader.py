import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from typing import Optional


class DataLoder:
    @staticmethod
    def load_diabetes_data(with_missing_values: Optional[bool] = False) -> pd.DataFrame:
        """
         Load the Diabetes dataset.

         Parameters:
         - with_missing_values (bool): If True, introduce missing values to the features.

         Returns:
         pd.DataFrame: DataFrame containing the feature and target variables.
         """
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

    @staticmethod
    def load_multi_target_df(with_missing_values: Optional[bool] = False):
        # Set random seed for reproducibility
        np.random.seed(0)

        # Generate synthetic regression dataset
        n_samples = 100  # Number of samples
        n_features = 5  # Number of features
        n_targets = 2  # Number of target columns

        # Generate features and targets
        X, y = make_regression(n_samples=n_samples, n_features=n_features + n_targets,
                               n_targets=n_targets, noise=0.1)

        # Separate features and targets
        features = X[:, :n_features]  # Extract the first 5 columns as features
        targets = X[:, n_features:]  # Extract the last 2 columns as targets

        # Create a DataFrame
        df = pd.DataFrame(np.hstack([features, targets]),
                          columns=[f"feature_{i}" for i in range(n_features)] + [f"target_{i}" for i in
                                                                                 range(n_targets)])

        if with_missing_values:
            # Introduce missing values randomly to the features
            features_columns = [f"feature_{i}" for i in range(n_features)]
            idx = np.random.choice(df.index, size=20, replace=False)
            df.loc[idx, features_columns] = np.nan

        return df
