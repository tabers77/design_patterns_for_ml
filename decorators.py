from abc import ABC
from time import time
from models import BaseModel
import logging
from typing import Any

logging.basicConfig(level=logging.INFO)


class TimerDecorator(BaseModel, ABC):
    def __init__(self, base_model):
        self._base_model = base_model

    def execute_pipeline_steps(
            self, data: Any, split_configs: Any, trainer_configs: Any, pipe_steps: Any = None
    ) -> Any:
        start_time = time()
        result = self._base_model.execute_pipeline_steps(data, split_configs, trainer_configs, pipe_steps)
        end_time = time()
        logging.info(f"Execution time: {end_time - start_time} seconds")
        return result
