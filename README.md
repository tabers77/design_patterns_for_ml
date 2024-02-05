# Design Patterns for Machine Learning (Baseline Models)

The purpose of this project is to demonstrate how design patterns from software engineering can be applied when building machine learning models. The methods developed here help train and evaluate baseline models in a more automated way. This project also serves as a template to build baseline models and start developing custom models from there. Note that this project is constantly evolving and adapting to different scenarios.

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Getting Started

### Installation

1. To get started, clone the repository:

   ```bash
   git clone https://github.com/tabers77/design_patterns_for_ml.git

2. Create your virtual environment and add it to .gitignore.

3. Install the requirements:

   ```bash 
   pip install -r requirements.txt

## Usage
0. Initialize the imports:

  ```python
import data_preprocessor as dp
import conf.config as cfg
import data_loader as dl
from models import ModelFactory

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import decorators as dec
   ```

1. Run test to test that current scripts work properly 

   ```bash 
   pytest tests.py
   
2. Initialize the necessary configurations:

  ```python
split_configs = cfg.SplitConfigs(target_col_name='target', train_size=0.80, cv=5, split_policy='x_y_splits_only')
trainer_configs = cfg.TrainerConfigs(preprocess_strategy='pipeline', custom_scoring=None,
                                     input_dim=5)  # {'neg_mean_squared_error': m.mean_squared_error}
   ```

3. Initialize the Data Loaders to load the dataset:

  ```python
df = dl.DataLoder().load_diabetes_data(with_missing_values=True)
preprocessed_df = dp.DataPreprocessor(df=df).execute_steps()
   ```

4. Create your model from the model factory:

  ```python
model_factory = ModelFactory()
model_pipe = model_factory.create_regressor_model(model_type='random_forest', trainer_configs=trainer_configs)
   ```

5. Execute pipeline steps:

  ```python
 results = model_pipe.execute_pipeline_steps(data=preprocessed_df,
                                             split_configs=split_configs,
                                             trainer_configs=trainer_configs,
                                            pipe_steps=[
                                                  ('imputer', SimpleImputer(strategy="median")),
                                                 ('scaler', StandardScaler())])

   ```

For more information on usage, refer to main.py


## Project Structure

design_patterns_for_ml
|-- conf
|-- .flake8
|-- .gitignore
|-- data_loader.py
|-- data_preprocessor.py
|-- decorators.py
|-- evaluators.py
|-- main.py
|-- model_builders.py
|-- models.py
|-- README.md
|-- requirements.txt
|-- tests.py
|-- trainer.py


