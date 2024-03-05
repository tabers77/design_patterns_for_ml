import conf.config as cfg
from models import ModelFactory, EvaluateEstimators
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
def data_loader() -> dl.DataLoder:
    """
    Fixture for creating a DataLoder instance.

    Returns:
    - dl.DataLoder: Instance of DataLoder.
    """
    return dl.DataLoder()


@pytest.fixture
def data_preprocessor(data_loader: dl.DataLoder) -> dp.DataPreprocessor:
    """
    Fixture for creating a DataPreprocessor instance.

    Parameters:
    - data_loader (dl.DataLoder): DataLoder instance.

    Returns:
    - dp.DataPreprocessor: Instance of DataPreprocessor.
    """
    df = data_loader.load_multi_target_df(with_missing_values=False)
    return dp.DataPreprocessor(df=df)


@pytest.fixture
def model_factory() -> ModelFactory:
    """
    Fixture for creating a ModelFactory instance.

    Returns:
    - ModelFactory: Instance of ModelFactory.
    """
    return ModelFactory()


# Adapted test cases based on variations
@pytest.mark.parametrize("split_policy", ['feature_target', 'x_y_splits_only'])
@pytest.mark.parametrize("target_col_names", ['target_0', ['target_0', 'target_1']])
@pytest.mark.parametrize("preprocess_strategy", ['custom', 'pipeline'])
@pytest.mark.parametrize("custom_scoring", [None, {'neg_mean_squared_error': m.mean_squared_error}])
@pytest.mark.parametrize("with_missing_values", [True, False])
def test_regression_models(data_preprocessor: dp.DataPreprocessor, model_factory: ModelFactory,
                           split_policy: str, target_col_names, preprocess_strategy: str,
                           custom_scoring: Optional[Dict[str, Any]],
                           with_missing_values: bool
                           ) -> None:
    """
    Test function for regression models.

    Parameters:
    - data_preprocessor (dp.DataPreprocessor): A DataPreprocessor instance.
    - model_factory (ModelFactory): A ModelFactory instance.
    - split_policy (str): Split policy for data splitting.
    - target_col_names: Target column names.
    - preprocess_strategy (str): Preprocessing strategy.
    - custom_scoring (Optional[Dict[str, Any]]): Custom scoring dictionary.
    - with_missing_values (bool): Flag indicating whether to include missing values.

    Returns:
    - None
    """
    # # Combinations exceptions - In case we need to exclude some combinations
    # if preprocess_strategy == 'pipeline' and custom_scoring is None:
    #     pytest.skip("Skipping test for combination of pipeline and x_y_splits_only without custom scoring")

    split_configs = cfg.SplitConfigs(target_col_names=target_col_names, train_size=0.80, cv=5,
                                     split_policy=split_policy)
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
                                                 trainer_configs=trainer_configs,
                                                 pipe_steps=pipe_steps)

    model_lr = model_factory.create_regressor_model(model_type='linear_regression',
                                                    trainer_configs=trainer_configs)

    results_lr = model_lr.execute_pipeline_steps(data=preprocessed_df,
                                                 trainer_configs=trainer_configs,
                                                 pipe_steps=pipe_steps)

    assert results_rf.mean_squared_error is not None
    assert results_lr.mean_squared_error is not None

    assert results_rf.results_table is not None
    assert results_lr.results_table is not None


@pytest.fixture
def train_test_comp(data_preprocessor: dp.DataPreprocessor) -> at.TrainVsTest:
    """
    Fixture for creating a TrainVsTest instance.

    Parameters:
    - data_preprocessor (dp.DataPreprocessor): DataPreprocessor instance.

    Returns:
    - at.TrainVsTest: Instance of TrainVsTest.
    """
    preprocessed_df = data_preprocessor.execute_steps()
    split_configs = cfg.SplitConfigs(target_col_names='target_0',
                                     train_size=0.80,
                                     cv=5,
                                     split_policy='train_test_splits_only')

    splits = DataSpliter(split_configs=split_configs, df=preprocessed_df).execute_split_steps()

    return at.TrainVsTest(train=splits.train, test=splits.test)


def test_get_covariance_shift_score(train_test_comp: at.TrainVsTest) -> None:
    """
    Test for the get_covariance_shift_score method.

    Parameters:
    - train_test_comp (at.TrainVsTest): Instance of TrainVsTest.

    Returns:
    - None
    """
    # TODO: TEST COVARIANCE WITH MULTIPLE TARGETS IN target_label, test_get_covariance_shift_score
    result = train_test_comp.get_covariance_shift_score(target_label='target_0')
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_get_covariance_shift_score_per_feature(train_test_comp: at.TrainVsTest) -> None:
    """
    Test for the get_covariance_shift_score_per_feature method.

    Parameters:
    - train_test_comp (at.TrainVsTest): Instance of TrainVsTest.

    Returns:
    - None
    """
    result = train_test_comp.get_covariance_shift_score_per_feature()
    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.parametrize("split_policy", ['feature_target', 'x_y_splits_only'])
@pytest.mark.parametrize("preprocess_strategy", ['custom', 'pipeline'])
@pytest.mark.parametrize("custom_scoring", [None, {'neg_mean_squared_error': m.mean_squared_error}])
@pytest.mark.parametrize("target_col_names", ['target_0', ['target_0', 'target_1']])
def test_evaluate_estimators(data_preprocessor: dp.DataPreprocessor, split_policy: str, preprocess_strategy: str,
                             custom_scoring: Optional[Dict[str, Any]], target_col_names) -> None:
    """
    Test for evaluating estimators.

    Parameters:
    - data_preprocessor (dp.DataPreprocessor): DataPreprocessor instance.
    - split_policy (str): Split policy for data splitting.
    - preprocess_strategy (str): Preprocessing strategy.
    - custom_scoring (Optional[Dict[str, Any]]): Custom scoring dictionary.
    - target_col_names: Target column names.

    Returns:
    - None
    """
    # Load and preprocess data

    preprocessed_df = data_preprocessor.execute_steps()

    # Set up configurations
    split_configs = cfg.SplitConfigs(target_col_names=target_col_names,
                                     train_size=0.80,
                                     cv=5,
                                     split_policy=split_policy)

    scorer = ut.Scorers(split_configs=split_configs, custom_scoring=custom_scoring)

    trainer_configs = cfg.TrainerConfigs(preprocess_strategy=preprocess_strategy,
                                         scorer=scorer,
                                         input_dim=5)

    # Initialize EvaluateEstimators object
    evaluator = EvaluateEstimators(df=preprocessed_df,
                                   trainer_configs=trainer_configs,
                                   regression=True,
                                   custom_estimators=None)  # TODO: ADD: [RandomForestModel(trainer_configs)]

    result = evaluator.execute_steps()

    # Ensure the result is of type DictResultHolder
    assert isinstance(result, ut.DictResultHolder)
