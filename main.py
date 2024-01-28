import data_preprocessor as dp
import conf.config as cfg
import data_loader as dl
from models import ModelFactory

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -------
# TASKS
# -------

# SPLIT METHODS NEEDS TO BE ADAPTABLE
# FINISH MLP
# CREATE A METHOD  TO THE BEST METHODS
# IMPLEMENT MULTI TARGET
# ADD AUTO MODE FUNCTIONALITY
# ADD DIFFERENT
split_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80)

if __name__ == '__main__':
    df = dl.DataLoder().load_data()
    preprocessed_df = dp.DataPreprocessor(configs=split_configs, df=df).execute_steps()
    model_factory = ModelFactory()
    model = model_factory.create_regressor_model(model_type='random_forest', preprocess_strategy='pipeline')
    results = model.execute_pipeline_steps(data=preprocessed_df,
                                           split_configs=split_configs,
                                           pipe_steps=[
                                               ('imputer', SimpleImputer(strategy="median")),
                                               ('scaler', StandardScaler())])
    print(results.mean_squared_error)
