import conf.config as cfg
from models import ModelFactory
import data_loader as dl
import data_preprocessor as dp
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as m
from typing import Optional, Dict, Any

TRAIN_SIZE = 0.80
CV_FOLDS = 5
IMPUTER_STRATEGY = "median"

@pytest.fixture
def data_loader():
    return dl.DataLoader()

@pytest.fixture
def data_preprocessor(data_loader):
    df = data_loader.load_diabetes_data(with_missing_values=False)
    return dp.DataPreprocessor(df=df)

@pytest.fixture
def model_factory():
    return ModelFactory()

@pytest.mark.parametrize("split_policy", ['feature_target', 'x_y_splits_only'])
@pytest.mark.parametrize("preprocess_strategy", ['custom', 'pipeline'])
@pytest.mark.parametrize("custom_scoring", [None, {'neg_mean_squared_error': m.mean_squared_error}])
@pytest.mark.parametrize("with_missing_values", [True, False])
def test_regression_models(data_preprocessor: dp.DataPreprocessor, model_factory: ModelFactory,
                           split_policy: str, preprocess_strategy: str, custom_scoring: Optional[Dict[str, Any]],
                           with_missing_values: bool) -> None:
    split_configs = cfg.SplitConfigs(target_col_name='target', train_size=TRAIN_SIZE, cv=CV_FOLDS, split_policy=split_policy)
    trainer_configs = cfg.TrainerConfigs(preprocess_strategy=preprocess_strategy, custom_scoring=custom_scoring,
                                         input_dim=5)

    if with_missing_values:
        pipe_steps = [('imputer', SimpleImputer(strategy=IMPUTER_STRATEGY)), ('scaler', StandardScaler())]
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

    # Additional tests for edge cases
    assert results_rf.mean_squared_error >= 0
    assert results_lr.mean_squared_error >= 0
    assert isinstance(results_rf.mean_squared_error, float)
    assert isinstance(results_lr.mean_squared_error, float)