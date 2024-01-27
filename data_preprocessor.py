from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)


class DataPreprocessor:
    # TODO: INDICATE target column name of multiple
    def __init__(self, configs, df):
        self.configs = configs
        self.df = df
        self.init_checks()

    def init_checks(self):
        # Check 1
        assert all(col in self.df.columns for col in ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6',
                                                      'target'])

        # IDEA: if there are missing columns indicate which ones , give a warning
        # print(self.df.isnull().sum())
        return self.df

    def standardization(self):
        pass

    def execute_steps(self):

        return self.init_checks()


class DataSpliter:
    def __init__(self, configs, df):
        self.configs = configs
        self.df = df
        self.target_col_name = self.configs.target_col_name
        self.train_size = self.configs.train_size

    # TODO: add different types of stratification
    def train_test_split(self):
        x, y = self.df.drop(self.target_col_name, axis=1), self.df[self.target_col_name]

        class_counts = y.value_counts()
        min_samples = class_counts.min()

        if min_samples < 2:
            logging.warning(f"The least populated class has only {min_samples} member(s), which is too few.")
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=self.train_size)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=self.train_size, stratify=y)

        return x_train, y_train, x_test, y_test

    def train_test_split_scaled(self):
        pass
