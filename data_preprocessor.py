# sanity checks
# preprocess data : check conditions, age < 0.038

class DataPreprocessor:
    # TODO: INDICATE target column name of multiple
    def __init__(self, configs, df, target_col_name):
        self.configs = configs
        self.df = df
        self.target_col_name = target_col_name
        self.init_checks()

    def init_checks(self):
        # Check 1
        assert all(col in self.df.columns for col in ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6',
                                                      'target'])

        # IDEA: if there are missing columns indicate which ones , give a warning
        print(self.df.isnull().sum())

    def train_test_split(self):
        x, y = self.df.drop(self.target_col_name, axis=1), self.df[self.target_col_name]
        # IDEA: HERE IT COULD BE STRATIFIED SAMPLE
        x_train, y_train = 9, 8
        x_test, y_test = 9, 8

    def train_test_split_scaled(self):
        # Use different scaler
        x_train_scaled, y_train_scaled = 9, 8
        x_test_scaled, y_test_scaled = 9, 8

    def standardization(self):
        pass

