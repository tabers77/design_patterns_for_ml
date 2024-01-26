import pandas as pd
from sklearn.datasets import load_diabetes


class DataLoder:
    @staticmethod
    def load_data():
        # Load Boston Housing dataset
        diabetes_df = load_diabetes()
        # Create a DataFrame with the feature and target variables
        data = pd.DataFrame(diabetes_df.data, columns=diabetes_df.feature_names)
        data['target'] = diabetes_df.target
        return data
