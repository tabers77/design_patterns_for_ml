from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import numpy as np
from scipy.stats import norm
from conf.config import Cfg
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)


class OutlierHandler:
    @staticmethod
    def is_distribution_normal(col: pd.Series):
        """Check if the distribution of a col is normal"""

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
    def get_outliers_std(df: pd.DataFrame, column: str):
        len_df = len(df)

        q25, q75 = np.percentile(df[column], 25), np.percentile(df[column], 75)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off

        outliers = [x for x in df[column] if x < lower or x > upper]
        pct_outliers = round(len(outliers) / len_df * 100, 2)

        return pct_outliers, outliers

    @staticmethod
    def get_outliers_z_score(df: pd.DataFrame, column: str):
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
    def identify_outliers_in_dataframe(df):
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
    # TODO: INDICATE target column name of multiple
    def __init__(self, df):
        self.df = df
        self.init_checks()

    def init_checks(self):
        # CHECK 2
        assert all(col in self.df.columns for col in Cfg.constants.fixed_columns)
        logging.info('CHECK #1: All required columns are present.')

        # CHECK 2
        logging.info(f'CHECK #2: Missing values check: {self.df.isnull().any().any()}')

        # CHECK 3
        outlier_handler = OutlierHandler()
        outliers_dict = outlier_handler.identify_outliers_in_dataframe(self.df)
        logging.info(f'CHECK #3: Outliers identified {outliers_dict}')

    def standardization(self):
        pass

    def execute_steps(self):
        return self.df


class DataSpliter:
    def __init__(self, configs, df):
        self.configs = configs
        self.df = df
        self.target_col_name = self.configs.target_col_name
        self.train_size = self.configs.train_size
        self.train_chunk = int(len(self.df) * self.train_size)

    # TODO: add different types of stratification
    def feature_target_split(self):
        x, y = self.df.drop(self.target_col_name, axis=1), self.df[self.target_col_name]

        class_counts = y.value_counts()
        min_samples = class_counts.min()

        if min_samples < 2:
            logging.warning(f"The least populated class has only {min_samples} member(s), which is too few.")
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=self.train_size)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=self.train_size, stratify=y)

        return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    def x_y_splits_only(self):
        x, y = self.df.drop(self.target_col_name, axis=1), self.df[self.target_col_name]
        return {'x': x, 'y': y}

    def train_test_splits_only(self):
        train, test = self.df[:self.train_chunk], self.df[self.train_chunk:]
        return {'train': train, 'test': test}

    def train_test_split_scaled(self):
        pass

    def execute_split_steps(self):
        if self.configs.split_policy == 'feature_target':
            return SplitResults(self.feature_target_split())
        elif self.configs.split_policy == 'x_y_splits_only':
            return SplitResults(self.x_y_splits_only())
        elif self.configs.split_policy == 'train_test_splits_only':
            return SplitResults(self.train_test_splits_only())
        else:
            raise ValueError(f"Unsupported split_policy: {self.configs.split_policy}")


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
