import data_preprocessor as dp
import conf.config as cfg
import data_loader as dl
from models import ModelFactory
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

split_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80, cv=5, split_policy='x_y_splits_only')

trainer_configs = cfg.TrainerConfigs(preprocess_strategy='pipeline', custom_scoring=None,
                                     input_dim=5)  # {'neg_mean_squared_error': m.mean_squared_error}

if __name__ == '__main__':
    df = dl.DataLoder().load_diabetes_data(with_missing_values=True)
    preprocessed_df = dp.DataPreprocessor(df=df).execute_steps()

    model_factory = ModelFactory()
    model_pipe = model_factory.create_regressor_model(model_type='random_forest', trainer_configs=trainer_configs)

    results = model_pipe.execute_pipeline_steps(data=preprocessed_df,
                                                split_configs=split_configs,
                                                trainer_configs=trainer_configs,
                                                pipe_steps=[
                                                    ('imputer', SimpleImputer(strategy="median")),
                                                    ('scaler', StandardScaler())])
    print(results)
