import data_preprocessor as dp
import conf.config as cfg
import data_loader as dl
from models import ModelFactory
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def main() -> None:
    split_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80, cv=5, split_policy='x_y_splits_only')
    trainer_configs = cfg.TrainerConfigs(preprocess_strategy='pipeline', custom_scoring=None, input_dim=5)

    data_loader = dl.DataLoader()
    df = data_loader.load_diabetes_data(with_missing_values=True)
    data_preprocessor = dp.DataPreprocessor(df=df)
    preprocessed_df = data_preprocessor.execute_steps()

    model_factory = ModelFactory()
    model_pipe = model_factory.create_regressor_model(model_type='random_forest', trainer_configs=trainer_configs)

    results = model_pipe.execute_pipeline_steps(data=preprocessed_df,
                                                split_configs=split_configs,
                                                trainer_configs=trainer_configs,
                                                pipe_steps=[
                                                    ('imputer', SimpleImputer(strategy="median")),
                                                    ('scaler', StandardScaler())])
    print(results)

if __name__ == '__main__':
    main()