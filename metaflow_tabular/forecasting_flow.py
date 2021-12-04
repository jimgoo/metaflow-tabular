"""
To run this flow:
```python forecasting_flow.py --environment=conda run```
"""

from functools import partial

from metaflow import (
    Flow,
    FlowSpec,
    IncludeFile,
    Parameter,
    batch,
    conda,
    conda_base,
    get_metadata,
    parallel_map,
    step,
)

from pip_decorator import pip
from forecasting_models import GluonTSModel, KatsModel, NeuralProphetModel, MerlionModel

# this version is used in pre and post processing steps
PANDAS_VERSION = "1.3.3"

# this version is used when conda packages aren't available
PIP_VERSION = "21.3.1"


def run_model(
    model_config, wrapper_class, target_index, forecast_steps, train_df, data_freq
):
    try:
        model = wrapper_class(
            model_config, target_index, forecast_steps, data_freq=data_freq
        )
        model.fit(train_df)
        forecast = model.predict(train_df)
        forecast["id"] = model_config["id"]
        return forecast
    except:
        print(f"Error with {model_config}")
        raise


@conda_base(python="3.8.12")
class ForecastingFlow(FlowSpec):
    """
    A flow for benchmarking forecasting libraries.
    """

    train_path = Parameter(
        "train_path",
        help="The path to a DataFrame file for training",
        default="https://jgoode.s3.amazonaws.com/ts-datasets/seattle-trail.csv",
    )

    test_path = Parameter(
        "test_path",
        help="The path to a DataFrame file for testing",
        default=None,
    )

    date_col = Parameter(
        "date_col",
        help="Column of the date in the input DataFrame",
        default="Date",
    )

    target_col = Parameter(
        "target_col",
        help="Column of the target in the input DataFrame",
        default="BGT North of NE 70th Total",
    )

    # data_config_path = Parameter(
    #     "data_config_path",
    #     help=

    model_config_path = Parameter(
        "model_config_path",
        help="The path to a model config file",
        default="../configs/forecasting/models/default.yaml",
    )

    forecast_steps = Parameter(
        "forecast_steps",
        help="The number of steps ahead to forecast",
        default=10,
    )

    @conda(libraries={"pandas": PANDAS_VERSION, "pyyaml": "6.0"})
    @step
    def start(self):
        """
        Start the flow by preprocessing the data.
        """
        import pandas as pd
        from pprint import pprint
        import yaml

        # Print the Metaflow metadata provider
        print(f"Using metadata provider: {get_metadata()}")

        def load_df(path):
            df = pd.read_csv(path)

            assert self.date_col in df.columns, '"%s" not in columns' % self.date_col
            assert self.target_col in df.columns, (
                '"%s" not in columns' % self.target_col
            )

            # parse date column and set it as the index
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df.set_index(self.date_col, inplace=True)
            return df

        self.train_df = load_df(self.train_path)
        if self.test_path is not None:
            self.test_df = load_df(self.test_path)
            assert (
                self.train_df.columns == self.test_df.columns
            ).all(), "Columns do not match"
        else:
            self.test_df = None

        if self.test_df is None:
            n_train = 500
            self.test_df = self.train_df.iloc[n_train : n_train + self.forecast_steps]
            self.train_df = self.train_df.iloc[:n_train]

        # get index of the target column
        self.target_index = self.train_df.columns.tolist().index(self.target_col)

        # get the frequency of the data
        self.freq = pd.infer_freq(self.train_df.index)

        # load the model config file
        with open(self.model_config_path, "r") as f:
            self.model_config = yaml.safe_load(f)

        print("train df")
        print(self.train_df)
        print("test df")
        print(self.test_df)
        print("model_config")
        pprint(self.model_config)

        # these branches will run in parallel
        # TODO: skip those with no entries in the model config
        self.next(
            self.run_merlion,
            self.run_gluonts,
            self.run_kats,
            self.run_neuralprophet,
        )

    @conda(libraries={"salesforce-merlion": "1.0.2"})
    @step
    def run_merlion(self):
        """
        Backtest Merlion models.
        https://github.com/salesforce/Merlion
        """
        self.forecasts = parallel_map(
            partial(
                run_model,
                wrapper_class=MerlionModel,
                target_index=self.target_index,
                forecast_steps=self.forecast_steps,
                train_df=self.train_df,
                data_freq=self.freq,
            ),
            self.model_config["libs"].get("merlion", []),
        )
        self.next(self.join)

    # We use pip because mxnet 1.5.0 is broken and there's no newer conda version.
    @pip(libraries={"mxnet": "1.8.0.post0", "gluonts": "0.8.1"})
    @conda(libraries={"pip": PIP_VERSION})
    @step
    def run_gluonts(self):
        """
        Backtest gluon-ts models.
        https://github.com/awslabs/gluon-ts
        """
        self.forecasts = parallel_map(
            partial(
                run_model,
                wrapper_class=GluonTSModel,
                target_index=self.target_index,
                forecast_steps=self.forecast_steps,
                train_df=self.train_df,
                data_freq=self.freq,
            ),
            self.model_config["libs"].get("gluonts", []),
        )
        self.next(self.join)

    @conda(libraries={"kats": "0.1.0"})
    @step
    def run_kats(self):
        """
        Backtest Kats models.
        https://github.com/facebookresearch/Kats
        """
        self.forecasts = parallel_map(
            partial(
                run_model,
                wrapper_class=KatsModel,
                target_index=self.target_index,
                forecast_steps=self.forecast_steps,
                train_df=self.train_df,
                data_freq=self.freq,
            ),
            self.model_config["libs"].get("kats", []),
        )
        self.next(self.join)

    # We use pip because there isn't a conda package for NeuralProphet.
    @pip(libraries={"neuralprophet": "0.3.0"})
    @conda(libraries={"pip": PIP_VERSION})
    @step
    def run_neuralprophet(self):
        """
        Backtest NeuralProphet models.
        https://github.com/ourownstory/neural_prophet
        """
        self.forecasts = parallel_map(
            partial(
                run_model,
                wrapper_class=NeuralProphetModel,
                target_index=self.target_index,
                forecast_steps=self.forecast_steps,
                train_df=self.train_df,
                data_freq=self.freq,
            ),
            self.model_config["libs"].get("neuralprophet", []),
        )
        self.next(self.join)

    @conda(libraries={"pandas": PANDAS_VERSION})
    @step
    def join(self, inputs):
        """
        Compute performance metrics for each library.
        """
        from collections import OrderedDict
        import numpy as np
        import pandas as pd

        forecasts = OrderedDict()

        # get forecasts for each library
        for lib in inputs:
            # carry these forward
            self.train_df = lib.train_df
            self.test_df = lib.test_df
            self.target_index = lib.target_index

            for forecast in lib.forecasts:
                assert (
                    forecast["id"] not in forecasts
                ), f"Duplicate forecast id: {forecast['id']}"
                forecasts[forecast["id"]] = forecast["y_hat"].reshape(-1)

        # get timestamps for the forecasts
        freq = self.train_df.index[1] - self.train_df.index[0]
        future_dates = pd.DatetimeIndex(
            [
                self.train_df.index[-1] + (i + 1) * freq
                for i in range(self.forecast_steps)
            ]
        )

        self.forecasts = pd.DataFrame(forecasts, index=future_dates)

        print("forecasts:")
        print(self.forecasts)

        if self.test_df is not None:
            # duplicate univariate target across columns for each model
            true = self.test_df.iloc[
                : self.forecast_steps, [self.target_index] * self.forecasts.shape[1]
            ]
            pred = self.forecasts

            print("--> true")
            print(true)
            print("--> pred")
            print(pred)

            self.rmse = pd.Series(
                np.sqrt(np.mean((pred.values - true.values) ** 2, axis=0)),
                index=self.forecasts.columns,
            ).sort_values()

            print(f"RMSE:")
            print(self.rmse)

        self.next(self.end)

    @step
    def end(self):
        """
        End of the flow
        """
        pass


if __name__ == "__main__":
    ForecastingFlow()
