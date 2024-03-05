from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from numpy import ndarray

from conf.config import Cfg
import sweetviz as sv
from scipy.stats import ttest_ind
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as m
import logging


def plot_correlation(df: pd.DataFrame) -> plt.Figure:
    """Plot correlation heatmap."""
    plt.figure(figsize=[12, 7])
    correlations = df.corr().round(2)
    sns.heatmap(correlations, annot=True, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
    return plt.gcf()


class TrainVsTest:
    """Class for comparing train and test datasets."""

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

    def get_report(self, target_label: str) -> None:
        """
        Generate train-test comparison and target analysis reports.

        Parameters:
            target_label (str): The target label column.

        Raises:
            ValueError: If categorical values are present in the dataset.
        """

        logging.info('Generating train test comparison report...')
        report_comp = sv.compare((self.train, 'x_train'), (self.test, 'x_test'))
        report_comp.show_html('Train_Test_Comparison.html')

        try:
            logging.info('Generating target analysis report...')
            target_comp = sv.compare(self.train, self.test, target_label)
            target_comp.show_html('Target_Analysis.html')

        except ValueError:
            logging.info('sweetviz does not support categorical values so we skip...')
            pass

    def get_train_test_counts(self, cardinality_limit: int = 20) -> None:
        """
        Plot bar charts of low cardinality columns' distributions in train and test datasets.

        Parameters:
            cardinality_limit (int): The maximum cardinality for a column to be considered low cardinality.
        """

        low_cardinality_cols = [cname for cname in self.train if self.train[cname].nunique() <= cardinality_limit and
                                self.train[cname].dtype == "object"]
        if len(low_cardinality_cols) > 0:
            for col in low_cardinality_cols:
                train_pct = self.train[col].value_counts() / len(self.train) * 100
                test_pct = self.test[col].value_counts() / len(self.test) * 100
                df_plot = pd.DataFrame([train_pct, test_pct])
                df_plot.index = ['train', 'test']
                df_plot = df_plot.transpose()
                df_plot = df_plot.reset_index().rename(columns={'index': 'col'})
                df_plot.plot.barh(x='col', y=['train', 'test'], title=f'{col}', cmap='coolwarm')

                plt.show()
        else:
            logging.warning('There are no low cardinality columns or dataset is not categorical')

    def is_distribution_different(self, alpha: float = 0.05) -> Tuple[pd.DataFrame, List[str]]:
        """
        Check if distributions of numerical columns in train and test datasets are different.

        Parameters:
            alpha (float): The significance level for the t-test.

        Returns:
            Tuple containing a DataFrame with p-values and a list of columns with significantly different distributions.
        """

        train_stats = self.train.describe().drop('count', axis=0)
        test_stats = self.test.describe().drop('count', axis=0)
        df = pd.DataFrame()
        num_cols = self.train.select_dtypes(exclude='object').columns
        diff_cols = []
        for col in num_cols:
            tscore, p_value = ttest_ind(self.train[col], self.test[col])

            if p_value < alpha:
                df[f'{col}_train'] = train_stats[col]
                df[f'{col}_test'] = test_stats[col]
                df[f'{col}_p_value'] = p_value
                diff_cols.append(col)

        if len(diff_cols) == 0:
            logging.info('All the the distributions from test set are similar to train set')
        else:
            logging.warning('There are columns with different distributions')

        return df, diff_cols

    def get_is_train_col(self, new_train: pd.DataFrame = None, new_test: pd.DataFrame = None,
                         target_label: str = None) -> pd.DataFrame:
        """
        Add a binary target column to the combined dataset.

        Parameters:
            new_train (pd.DataFrame): Optional new training data.
            new_test (pd.DataFrame): Optional new test data.
            target_label (str): The target label column to drop.

        Returns:
            DataFrame with added 'is_train' column.
        """

        train = self.train.copy() if new_train is None else new_train
        test = self.test.copy() if new_test is None else new_test
        train.loc[:, 'is_train'] = 1
        test.loc[:, 'is_train'] = 0
        dataframe = pd.concat([train, test])
        dataframe['is_train'] = dataframe['is_train'].apply(lambda x: 1 if x == 1.0 else 0)
        if target_label is not None:
            dataframe.drop(target_label, axis=1, inplace=True)

        # TODO: ADD ENCODING
        # dataframe = enc.default_encoding(dataframe)

        return dataframe

    def train_test_pairplot(self, diag_kind: str = "hist") -> None:
        """
        Plot pairplot of features to visualize distribution differences between train and test datasets.

        Parameters:
            diag_kind (str): Type of diagonal plots. Supported types are 'hist' and 'kde'.
        """
        df, diff_cols = self.is_distribution_different()

        if len(diff_cols) > 1:
            diff_cols.append('is_train')
            full_data = self.get_is_train_col()
            sns.pairplot(full_data[diff_cols], hue='is_train', diag_kind=diag_kind)
            plt.show()
        else:
            logging.info('All the the distributions from test set are similar to train set')

    @staticmethod
    def plot_distribution_pairs(train: pd.DataFrame, test: pd.DataFrame, feature: str,
                                dist_plot_type: str = 'hist_plot', hue: str = "set", palette: List[str] = None,
                                ax: plt.Axes = None) -> plt.Axes:
        """
        Plot distribution pairs of a feature in train and test datasets.

        Parameters:
            train (pd.DataFrame): DataFrame containing training data.
            test (pd.DataFrame): DataFrame containing test data.
            feature (str): The feature column to plot.
            dist_plot_type (str): Type of distribution plot. Supported types are 'hist_plot' and 'kde_plot'.
            hue (str): Variable in the data to map plot aspects to different colors.
            palette (List[str]): List of colors to use for the plot elements.
            ax (plt.Axes): Axes object to draw the plot onto; otherwise uses the current Axes.

        Returns:
            plt.Axes: The Axes object containing the plot.
        """
        data_df = train.copy()
        data_df['set'] = 'train'
        data_df = pd.concat([data_df, test.copy()]).fillna('test')
        data_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        for i, s in enumerate(data_df[hue].unique()):
            selection = data_df.loc[data_df[hue] == s, feature]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                if dist_plot_type == 'kde_plot':
                    g = sns.kdeplot(selection, color=palette[i], ax=ax, label=s)

                elif dist_plot_type == 'hist_plot':
                    g = sns.histplot(selection, color=palette[i], ax=ax, label=s)
                else:
                    raise ValueError('dist_plot_type is not recognized')

        ax.set_title(f"Paired train/test distributions of {feature}")
        g.legend()
        return ax

    @staticmethod
    def plot_distribution_pairs_boxplot(train: pd.DataFrame, test: pd.DataFrame, feature: str, hue: str = "set",
                                        palette: List[str] = None, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot boxplot of a feature in train and test datasets.

        Parameters:
            train (pd.DataFrame): DataFrame containing training data.
            test (pd.DataFrame): DataFrame containing test data.
            feature (str): The feature column to plot.
            hue (str): Variable in the data to map plot aspects to different colors.
            palette (List[str]): List of colors to use for the plot elements.
            ax (plt.Axes): Axes object to draw the plot onto; otherwise uses the current Axes.

        Returns:
            plt.Axes: The Axes object containing the plot.
        """
        data_df = train.copy()
        data_df['set'] = 'train'
        data_df = pd.concat([data_df, test.copy()]).fillna('test')
        data_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            # g = sns.boxplot(x=hue, y=feature, data=data_df, palette=palette, ax=ax)

        # Set labels for the legend
        # We should set the legend once, not for each unique value of hue
        if ax is not None:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, data_df[hue].unique(), title=hue)

        ax.set_title(f"Paired train/test distributions of {feature}")
        return ax

    def plot_distribution_pairs_wrapper(self, train: pd.DataFrame, test: pd.DataFrame, use_boxplot: bool = False,
                                        dist_plot_type: str = 'hist_plot', selected_columns: List[str] = None,
                                        max_n_features: int = None) -> plt.Figure:
        """
        Plot distribution pairs of features in train and test datasets.

        Parameters:
            train (pd.DataFrame): DataFrame containing training data.
            test (pd.DataFrame): DataFrame containing test data.
            use_boxplot (bool): Whether to use boxplot for distribution visualization.
            dist_plot_type (str): Type of distribution plot. Supported types are 'hist_plot' and 'kde_plot'.
            selected_columns (List[str]): List of feature columns to plot.
            max_n_features (int): Maximum number of features to plot.

        Returns:
            plt.Figure: The Figure object containing the plot.
        """
        if train.isnull().sum().any() or train.isnull().sum().any():
            raise ValueError('The function does not support datasets with missing values')

        if selected_columns is None:
            numeric_features = train.select_dtypes([int, float]).columns if max_n_features is None else \
                train.select_dtypes([int, float]).columns[:max_n_features]
        else:
            numeric_features = selected_columns

        # Create a grid for subplots based on the number of features
        num_features = len(numeric_features)
        cols = max_n_features if max_n_features <= 4 else 4
        rows = (num_features + cols - 1) // cols

        # Create a figure and axis objects
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()  # Flatten axes if more than 1 row

        # Iterate through each feature and plot its distribution in a separate subplot
        for i, feature in enumerate(numeric_features):
            if use_boxplot:

                self.plot_distribution_pairs_boxplot(train, test, feature, palette=Cfg.constants.color_list, ax=axes[i])
            else:
                self.plot_distribution_pairs(train,
                                             test,
                                             feature,
                                             dist_plot_type=dist_plot_type,
                                             palette=Cfg.constants.color_list,
                                             ax=axes[i])

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        # plt.legend()

        # Observe that this will create an interactive window that remains open until closed by the user
        plt.show()

        return fig

    def get_covariance_shift_score(self, target_label: str = None,
                                   estimator: RandomForestClassifier = RandomForestClassifier(max_depth=2),
                                   n_folds: int = 5) -> Tuple[ndarray, ndarray]:
        """
        Compute covariance shift score based on AUC-ROC.

        Parameters:
            target_label (str): The target label column.
            estimator (RandomForestClassifier): The classifier for computing AUC-ROC.
            n_folds (int): Number of folds for cross-validation.

        - https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/
        Note: if the value of AUC-ROC for a particular feature is greater than 0.80, we classify that feature as
        drifting.

        Returns:
            Tuple containing mean and standard deviation of covariance shift scores.
        """

        size = int(len(self.test))

        folds_lst = [i for i in range(0, len(self.train) + 1, size)]
        logging.info(f'Current selected folds: {folds_lst}')
        cov_scores = []

        for fold in range(0, len(folds_lst) - 1):
            # Using the fold list we pick start and end value to have similar shape for train and test
            # Ex: if fold_lst is [0, 2000, 4000, 6000, 8000] start value will start from 0 to 2000 and
            # 2000 to 4000 and so on
            new_train = self.train[folds_lst[fold]:folds_lst[fold + 1]].copy()
            logging.info(f'new_train shape: {new_train.shape}')

            full_data = self.get_is_train_col(new_train=new_train, target_label=target_label)
            x, y = full_data.drop('is_train', axis=1), full_data['is_train']

            roc_auc_scorer = m.make_scorer(m.roc_auc_score)
            scores = ms.cross_val_score(estimator, x, y, cv=n_folds,
                                        scoring=roc_auc_scorer, n_jobs=-1, verbose=0)

            logging.info(f'Score for fold {fold}: {scores}')
            cov_scores.append(scores[0])

        if np.mean(cov_scores) >= 0.80:
            logging.warning('There are drifting features in your dataset, Please investigate this!!')

        logging.info(f'Mean score: {np.mean(cov_scores)}, Standard deviation: {np.std(cov_scores)}')
        return np.mean(cov_scores), np.std(cov_scores)

    def get_covariance_shift_score_per_feature(self,
                                               estimator: RandomForestClassifier = RandomForestClassifier(max_depth=2),
                                               cov_score_thresh: float = 0.80, n_folds: int = 5) \
            -> Tuple[dict, List[str]]:
        """
        Compute covariance shift scores per feature based on AUC-ROC.

        Parameters:
            estimator (RandomForestClassifier): The classifier for computing AUC-ROC.
            cov_score_thresh (float): Threshold for classifying features as drifting.
            n_folds (int): Number of folds for cross-validation.

        Returns:
            Tuple containing a dictionary with mean and standard deviation of covariance shift scores per feature,
            and a list of features classified as drifting.
        """

        size = int(len(self.test))
        folds_lst = [i for i in range(0, len(self.train) + 1, size)]
        logging.info(f'Current selected folds: {folds_lst}')
        cov_scores = {}

        for fold in range(0, len(folds_lst) - 1):
            # Using the fold list we pick start and end value to have similar shape for train and test
            # Ex: if fold_lst is [0, 2000, 4000, 6000, 8000] start value will start from 0 to 2000 and
            # 2000 to 4000 and so on
            new_train = self.train[folds_lst[fold]:folds_lst[fold + 1]]
            logging.info(f'new_train shape: {new_train.shape}')
            full_data = self.get_is_train_col(new_train=new_train, target_label=None)
            x, y = full_data.drop('is_train', axis=1), full_data['is_train']

            for col in x.columns:
                roc_auc_scorer = m.make_scorer(m.roc_auc_score)
                scores = ms.cross_val_score(estimator, x[[col]], y, cv=n_folds,
                                            scoring=roc_auc_scorer, n_jobs=-1, verbose=0)

                logging.info(f'Score for {col} in fold {fold}: {scores}')

                cov_scores.setdefault(f'{col}', []).append(scores[0])

        cov_scores = {k: (np.mean(v), np.std(v)) for k, v in cov_scores.items()}

        drop_list = [k for k, v in cov_scores.items() if v[0] > cov_score_thresh]

        return cov_scores, drop_list
