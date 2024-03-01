import conf.config as cfg
from models import ModelFactory
import data_loader as dl
import data_preprocessor as dp
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as m
from typing import Optional, Dict, Any
import analytics_tools as at
from data_preprocessor import DataSpliter
import utils as ut


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
def test_regression_models(data_preprocessor: dp.DataPreprocessor, model_factory: ModelFactory,
                           split_policy: str, preprocess_strategy: str, custom_scoring: Optional[Dict[str, Any]],
                           with_missing_values: bool) -> None:
    """
    Test function for regression models.

    Parameters:
    - data_preprocessor: A DataPreprocessor instance.
    - model_factory: A ModelFactory instance.
    - split_policy: Split policy for data splitting.
    - preprocess_strategy: Preprocessing strategy.
    - custom_scoring: Custom scoring dictionary.
    - with_missing_values: Flag indicating whether to include missing values.

    Returns:
    - None
    """
    # # Combinations exceptions - In case we need to exclude some combinations
    # if preprocess_strategy == 'pipeline' and custom_scoring is None:
    #     pytest.skip("Skipping test for combination of pipeline and x_y_splits_only without custom scoring")

    split_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80, cv=5, split_policy=split_policy)
    scorer = ut.Scorers(split_configs=split_configs, custom_scoring=custom_scoring)

    trainer_configs = cfg.TrainerConfigs(preprocess_strategy=preprocess_strategy, scorer=scorer,
                                         input_dim=5)
    if with_missing_values:
        pipe_steps = [('imputer', SimpleImputer(strategy="median")), ('scaler', StandardScaler())]
    else:
        pipe_steps = None

    preprocessed_df = data_preprocessor.execute_steps()

    model_rf = model_factory.create_regressor_model(model_type='random_forest', trainer_configs=trainer_configs)
    results_rf = model_rf.execute_pipeline_steps(data=preprocessed_df,
                                                 split_configs=split_configs,
                                                 trainer_configs=trainer_configs,
                                                 pipe_steps=pipe_steps)

    model_lr = model_factory.create_regressor_model(model_type='linear_regression',
                                                    trainer_configs=trainer_configs)

    results_lr = model_lr.execute_pipeline_steps(data=preprocessed_df,
                                                 split_configs=split_configs,
                                                 trainer_configs=trainer_configs,
                                                 pipe_steps=pipe_steps)

    assert results_rf.mean_squared_error is not None
    assert results_lr.mean_squared_error is not None


@pytest.fixture
def train_test_comp(data_preprocessor: dp.DataPreprocessor):
    preprocessed_df = data_preprocessor.execute_steps()
    split_configs = cfg.SplitConfigs(target_col_name='target',
                                     train_size=0.80,
                                     cv=5,
                                     split_policy='train_test_splits_only')

    splits = DataSpliter(configs=split_configs, df=preprocessed_df).execute_split_steps()

    return at.TrainVsTest(train=splits.train, test=splits.test)


def test_get_covariance_shift_score(train_test_comp):
    result = train_test_comp.get_covariance_shift_score(target_label='target')
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_get_covariance_shift_score_per_feature(train_test_comp):
    result = train_test_comp.get_covariance_shift_score_per_feature()
    assert isinstance(result, tuple)
    assert len(result) == 2
