import data_preprocessor as dp
import conf.config as cfg
import data_loader as dl
from models import ModelFactory


custom_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80)


if __name__ == '__main__':
    df = dl.DataLoder().load_data()
    preprocessed_df = dp.DataPreprocessor(configs=custom_configs, df=df).execute_steps()
    model_factory = ModelFactory()
    model = model_factory.create_regressor_model(model_type='random_forest')
    results = model.execute_pipeline_steps(data=preprocessed_df, configs=custom_configs)
    print(results.results_table)