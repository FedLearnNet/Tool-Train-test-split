from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic.dataclasses import dataclass
from pyfedappwrap.learning.app_types import BasePrePostProcessApp
from pyfedappwrap.learning.run_runfig import AppConfig, AppOutputConfig, AppInputConfig
from sklearn.model_selection import train_test_split


@dataclass
class DataSplittingConfig(AppConfig):
    test_size: float | int = 0.25
    random_state: int = 42
    shuffle: bool = True


@dataclass
class DataSplittingOutput(AppOutputConfig):
    train: Any | None = None
    test: Any | None = None


@dataclass
class DataSplittingInput(AppInputConfig):
    input: Any = None


class DataSplittingApp(
    BasePrePostProcessApp[DataSplittingConfig, DataSplittingInput, DataSplittingOutput]):

    def __init__(self):
        super().__init__()

    def run_process(self, data: DataSplittingInput) -> DataSplittingOutput:
        if data.input is None:
            raise ValueError("DataSplittingInput.input must be a pandas DataFrame, got None.")
        if not isinstance(data.input, pd.DataFrame):
            raise TypeError("DataSplittingInput.input must be a pandas DataFrame.")

        test_size = self.config.test_size
        if test_size is None:
            test_size = 0.25

        train_df, test_df = train_test_split(
            data.input,
            test_size=test_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle,
        )

        return DataSplittingOutput(train=train_df, test=test_df)
