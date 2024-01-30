import data_preprocessor as dp
import conf.config as cfg
import data_loader as dl
from models import ModelFactory

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# --------------
# TASKS STAGE 1
# --------------
# IMPLEMENT MODEL TRAINER
# FINISH MLP
# ADD UTILS ?
# FINAL REFACTOR STAGE 1

# --------------
# TASKS STAGE 2
# --------------
# CREATE A METHOD  TO THE BEST METHODS
# IMPLEMENT MULTI TARGET
# Hyper parameter tuning
# MLFlow
# ADD AUTO MODE FUNCTIONALITY
# Unit tests


split_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80, split_policy='feature_target')

if __name__ == '__main__':
    df = dl.DataLoder().load_diabetes_data(with_missing_values=False)
    preprocessed_df = dp.DataPreprocessor(df=df).execute_steps()

    model_factory = ModelFactory()
    model = model_factory.create_regressor_model(model_type='random_forest', preprocess_strategy='custom')
    results = model.execute_pipeline_steps(data=preprocessed_df,
                                           split_configs=split_configs,
                                           pipe_steps=[
                                               ('imputer', SimpleImputer(strategy="median")),
                                               ('scaler', StandardScaler())])
    print(results.mean_squared_error)
