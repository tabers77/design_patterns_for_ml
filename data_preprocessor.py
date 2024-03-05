from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import numpy as np
from scipy.stats import norm
import conf.config as conf
from typing import Dict, Any, Tuple, List

logging.basicConfig(level=logging.INFO)


class OutlierHandler:
    @staticmethod
    def is_distribution_normal(col: pd.Series) -> bool:
        """
        Check if the distribution of a column is normal.

        Parameters:
        - col (pd.Series): The input column.

        Returns:
        bool: True if the distribution is normal, False otherwise.
        """

        mean = col.mean()
        sd = col.std()

        one_sd = norm.cdf(sd, mean, sd) - norm.cdf(-sd, mean, sd)
        two_sd = norm.cdf(2 * sd, mean, sd) - norm.cdf(-2 * sd, mean, sd)
        three_sd = norm.cdf(3 * sd, mean, sd) - norm.cdf(-3 * sd, mean, sd)

        counter = 0

        if 0.68 <= one_sd < 0.69:
            counter += 1

        if 0.95 <= two_sd < 0.96:
            counter += 1

        if 0.99 <= three_sd < 1:
            counter += 1

        if counter == 3:
            return True
        else:
            return False

    @staticmethod
    def get_outliers_std(df: pd.DataFrame, column: str) -> Tuple[float, List[float]]:
        """
        Get the percentage of outliers and the list of outliers using the standard deviation method.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - column (str): The column name.

        Returns:
        Tuple[float, List[float]]: The percentage of outliers and the list of outliers.
        """
        len_df = len(df)

        q25, q75 = np.percentile(df[column], 25), np.percentile(df[column], 75)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off

        outliers = [x for x in df[column] if x < lower or x > upper]
        pct_outliers = round(len(outliers) / len_df * 100, 2)

        return pct_outliers, outliers

    @staticmethod
    def get_outliers_z_score(df: pd.DataFrame, column: str) -> Tuple[float, List[float]]:
        """
        Get the percentage of outliers and the list of outliers using the Z-score method.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - column (str): The column name.

        Returns:
        Tuple[float, List[float]]: The percentage of outliers and the list of outliers.
        """
        len_df = len(df)
        outliers = []
        threshold = 3
        mean = np.mean(df[column])
        std = np.std(df[column])
        for i in df[column]:
            z_score = (i - mean) / std
            if np.abs(z_score) > threshold:
                outliers.append(i)
        # print(f'Identified outliers for {column}: {round(len(outliers) / len_df * 100, 2)}%')
        pct_outliers = round(len(outliers) / len_df * 100, 2)

        return pct_outliers, outliers

    @staticmethod
    def identify_outliers_in_dataframe(df: pd.DataFrame) -> Dict[str, Tuple[float, List[float]]]:
        """
        Identify outliers in a DataFrame for each column.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.

        Returns:
        Dict[str, Tuple[float, List[float]]]: A dictionary containing column names and corresponding
        percentages of outliers and lists of outliers.
        """
        outliers_dict = dict()
        outlier_handler = OutlierHandler()
        for col in df.columns:
            if not outlier_handler.is_distribution_normal(df[col]):
                pct_outliers, outliers = outlier_handler.get_outliers_std(df, col)
            else:
                pct_outliers, outliers = outlier_handler.get_outliers_z_score(df, col)

            outliers_dict[col] = (pct_outliers, outliers)

        return outliers_dict


class DataPreprocessor:
    def __init__(self, df, fixed_columns=None):
        self.df = df
        self.fixed_columns = fixed_columns
        self.init_checks()

    def init_checks(self) -> None:
        """
        Perform initialization checks on the DataFrame.

        Returns:
        None
        """
        # CHECK 1
        if self.fixed_columns is not None:
            assert all(col in self.df.columns for col in self.fixed_columns)
            logging.info('CHECK #1: All required columns are present.')

        # CHECK 2
        logging.info(f'CHECK #2: Missing values check: {self.df.isnull().any().any()}')

        # CHECK 3
        outlier_handler = OutlierHandler()
        outliers_dict = outlier_handler.identify_outliers_in_dataframe(self.df)
        logging.info(f'CHECK #3: Outliers identified {outliers_dict}')

    def standardization(self):
        raise NotImplementedError

    def execute_steps(self):
        return self.df


class DataSpliter:
    def __init__(self, split_configs: conf.SplitConfigs, df: pd.DataFrame):
        """
        Initialize the DataSplitter with configuration and a DataFrame.

        Parameters:
        - configs (SplitConfigs): The configuration object.
        - df (pd.DataFrame): The input DataFrame.
        """
        self.split_configs: conf.SplitConfigs = split_configs
        self.df: pd.DataFrame = df
        # self.target_col_name: str = self.split_configs.target_col_name
        self.target_col_names: str = self.split_configs.target_col_names  # TEST
        self.train_size: float = self.split_configs.train_size
        self.train_chunk: int = int(len(self.df) * self.train_size)

    def get_feature_targets(self):
        if len(self.target_col_names) == 0:
            raise ValueError('The list of target column names cannot be empty. '
                             'Please provide at least one target column name.')
        features = self.df.drop(self.target_col_names, axis=1)
        targets = self.df[self.target_col_names]
        return features, targets

    # TODO: add different types of stratification
    def feature_target_split(self) -> Dict[str, pd.DataFrame]:
        """
        Perform feature-target split on the DataFrame.

        Returns:
        Dict[str, pd.DataFrame]: A dictionary containing splits for features and target.
        """

        x, y = self.get_feature_targets()

        if len(self.target_col_names) > 1:

            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=self.train_size)

        else:

            class_counts = y.value_counts()
            min_samples = class_counts.min()

            if min_samples < 2:
                logging.warning(f"The least populated class has only {min_samples} member(s), which is too few.")
                x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=self.train_size)
            else:
                x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=self.train_size, stratify=y)

        return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    def x_y_splits_only(self) -> Dict[str, pd.DataFrame]:
        """
        Perform feature-target split on the DataFrame and return only X and Y.

        Returns:
        Dict[str, pd.DataFrame]: A dictionary containing splits for features (X) and target (Y).
        """

        x, y = self.get_feature_targets()

        return {'x': x, 'y': y}

    def train_test_splits_only(self) -> Dict[str, pd.DataFrame]:
        """
        Perform train-test split on the DataFrame.

        Returns:
        Dict[str, pd.DataFrame]: A dictionary containing splits for training and testing.
        """
        train, test = self.df[:self.train_chunk], self.df[self.train_chunk:]
        return {'train': train, 'test': test}

    def train_test_split_scaled(self):
        pass

    def execute_split_steps(self) -> 'SplitResults':
        """
        Execute splitting steps based on the configured policy.

        Returns:
        SplitResults: The result of the split.
        """
        if self.split_configs.split_policy == 'feature_target':
            return SplitResults(self.feature_target_split())
        elif self.split_configs.split_policy == 'x_y_splits_only':
            return SplitResults(self.x_y_splits_only())
        elif self.split_configs.split_policy == 'train_test_splits_only':
            return SplitResults(self.train_test_splits_only())
        else:
            raise ValueError(f"Unsupported split_policy: {self.split_configs.split_policy}")


class SplitResults:
    """
    Class for holding split results.
    """

    def __init__(self, splits: Dict[str, Any]):
        """
        Initialize the SplitResults with a dictionary of split results.

        Parameters:
        - splits: A dictionary containing split results.
        """
        for split_name, split_value in splits.items():
            setattr(self, split_name, split_value)
