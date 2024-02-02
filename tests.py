
import conf.config as cfg
from models import ModelFactory
import data_loader as dl
import data_preprocessor as dp
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as m

@pytest.fixture
def data_loader():
    return dl.DataLoder()


@pytest.fixture
def data_preprocessor(data_loader):
    df = data_loader.load_diabetes_data(with_missing_values=False)
    return dp.DataPreprocessor(df=df)


@pytest.fixture
def model_factory():
    return ModelFactory()


# Adapted test cases based on variations
@pytest.mark.parametrize("split_policy", ['feature_target', 'x_y_splits_only'])
@pytest.mark.parametrize("preprocess_strategy", ['custom', 'pipeline'])
@pytest.mark.parametrize("custom_scoring", [None, {'neg_mean_squared_error': m.mean_squared_error}])
@pytest.mark.parametrize("with_missing_values", [True, False])
def test_regression_models(data_preprocessor, model_factory, split_policy, preprocess_strategy, custom_scoring,
                           with_missing_values):
    split_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80, cv=5, split_policy=split_policy)
    trainer_configs = cfg.TrainerConfigs(preprocess_strategy=preprocess_strategy, custom_scoring=custom_scoring,
                                         input_dim=5)

    if with_missing_values:
        pipe_steps = [('imputer', SimpleImputer(strategy="median")), ('scaler', StandardScaler())]
    else:
        pipe_steps = None

    preprocessed_df = data_preprocessor.execute_steps()

    model_rf = model_factory.create_regressor_model(model_type='random_forest', trainer_configs=trainer_configs)
    results_rf = model_rf.execute_pipeline_steps(data=preprocessed_df, split_configs=split_configs,
                                                 trainer_configs=trainer_configs, pipe_steps=pipe_steps)

    model_lr = model_factory.create_regressor_model(model_type='linear_regression', trainer_configs=trainer_configs)
    results_lr = model_lr.execute_pipeline_steps(data=preprocessed_df, split_configs=split_configs,
                                                 trainer_configs=trainer_configs, pipe_steps=pipe_steps)

    assert results_rf.mean_squared_error is not None
    assert results_lr.mean_squared_error is not None



# @pytest.fixture
# def data_loader():
#     return dl.DataLoder()
#
#
# @pytest.fixture
# def data_preprocessor(data_loader):
#     df = data_loader.load_diabetes_data(with_missing_values=False)
#     return dp.DataPreprocessor(df=df)
#
#
# @pytest.fixture
# def model_factory():
#     return ModelFactory()
#
#
# def test_feature_target_custom_two_models(data_preprocessor, model_factory):
#     split_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80, cv=5, split_policy='feature_target')
#     trainer_configs = cfg.TrainerConfigs(preprocess_strategy='custom')
#
#     preprocessed_df = data_preprocessor.execute_steps()
#
#     model_rf = model_factory.create_regressor_model(model_type='random_forest', trainer_configs=trainer_configs)
#     results_rf = model_rf.execute_pipeline_steps(data=preprocessed_df, split_configs=split_configs,
#                                                  trainer_configs=trainer_configs)
#
#     model_lr = model_factory.create_regressor_model(model_type='linear_regression', trainer_configs=trainer_configs)
#     results_lr = model_lr.execute_pipeline_steps(data=preprocessed_df, split_configs=split_configs,
#                                                  trainer_configs=trainer_configs)
#
#     assert results_rf.mean_squared_error is not None
#     assert results_lr.mean_squared_error is not None
#
#
# def test_x_y_splits_only_custom_two_models(data_preprocessor, model_factory):
#     split_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80, cv=5, split_policy='x_y_splits_only')
#     trainer_configs = cfg.TrainerConfigs(preprocess_strategy='custom')
#
#     preprocessed_df = data_preprocessor.execute_steps()
#
#     model_rf = model_factory.create_regressor_model(model_type='random_forest', trainer_configs=trainer_configs)
#     results_rf = model_rf.execute_pipeline_steps(data=preprocessed_df, split_configs=split_configs,
#                                                  trainer_configs=trainer_configs)
#
#     model_lr = model_factory.create_regressor_model(model_type='linear_regression', trainer_configs=trainer_configs)
#     results_lr = model_lr.execute_pipeline_steps(data=preprocessed_df, split_configs=split_configs,
#                                                  trainer_configs=trainer_configs)
#
#     assert results_rf.mean_squared_error is not None
#     assert results_lr.mean_squared_error is not None
#
#
# def test_x_y_splits_only_pipeline_two_models(data_preprocessor, model_factory):
#     split_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80, cv=5, split_policy='x_y_splits_only')
#     trainer_configs = cfg.TrainerConfigs(preprocess_strategy='pipeline')
#
#     preprocessed_df = data_preprocessor.execute_steps()
#
#     model_rf = model_factory.create_regressor_model(model_type='random_forest', trainer_configs=trainer_configs)
#     results_rf = model_rf.execute_pipeline_steps(data=preprocessed_df, split_configs=split_configs,
#                                                  trainer_configs=trainer_configs,
#                                                  pipe_steps=[
#                                                      ('imputer', SimpleImputer(strategy="median")),
#                                                      ('scaler', StandardScaler())])
#
#     model_lr = model_factory.create_regressor_model(model_type='linear_regression', trainer_configs=trainer_configs)
#     results_lr = model_lr.execute_pipeline_steps(data=preprocessed_df, split_configs=split_configs,
#                                                  trainer_configs=trainer_configs,
#                                                  pipe_steps=[
#                                                      ('imputer', SimpleImputer(strategy="median")),
#                                                      ('scaler', StandardScaler())])
#
#     assert results_rf.mean_squared_error is not None
#     assert results_lr.mean_squared_error is not None
#
#
# def test_feature_target_pipeline_two_models(data_preprocessor, model_factory):
#     split_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80, cv=5, split_policy='feature_target')
#     trainer_configs = cfg.TrainerConfigs(preprocess_strategy='pipeline')
#
#     preprocessed_df = data_preprocessor.execute_steps()
#
#     model_rf = model_factory.create_regressor_model(model_type='random_forest', trainer_configs=trainer_configs)
#     results_rf = model_rf.execute_pipeline_steps(data=preprocessed_df, split_configs=split_configs,
#                                                  trainer_configs=trainer_configs,
#                                                  pipe_steps=[
#                                                      ('imputer', SimpleImputer(strategy="median")),
#                                                      ('scaler', StandardScaler())])
#
#     model_lr = model_factory.create_regressor_model(model_type='linear_regression', trainer_configs=trainer_configs)
#     results_lr = model_lr.execute_pipeline_steps(data=preprocessed_df, split_configs=split_configs,
#                                                  trainer_configs=trainer_configs,
#                                                  pipe_steps=[
#                                                      ('imputer', SimpleImputer(strategy="median")),
#                                                      ('scaler', StandardScaler())])
#
#     assert results_rf.mean_squared_error is not None
#     assert results_lr.mean_squared_error is not None
