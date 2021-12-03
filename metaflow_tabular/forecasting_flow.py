"""
To run this flow:
```python forecasting_flow.py --environment=conda run```
"""

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

# use this version in pre and post processing steps
PANDAS_VERSION = "1.3.3"

# installed for all steps
# not required but helpful for debugging with jupyter
IPYKERNEL_VERSION = "6.5.0"

# for when we must use pip instead of conda
PIP_VERSION = "21.3.1"


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

    @conda(
        libraries={
            "ipykernel": IPYKERNEL_VERSION,
            "pandas": PANDAS_VERSION,
            "pyyaml": "6.0",
        }
    )
    @step
    def start(self):
        """
        Start the flow by preprocessing the data.
        """
        from io import StringIO
        from pprint import pprint

        import pandas as pd
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

        # branches will run in parallel
        self.next(
            self.backtest_merlion,
            self.backtest_gluonts,
            self.backtest_kats,
            self.backtest_neuralprophet,
        )

    # @batch(cpu=2, memory=4000)
    @conda(libraries={"ipykernel": IPYKERNEL_VERSION, "salesforce-merlion": "1.0.2"})
    @step
    def backtest_merlion(self):
        """
        Backtest Merlion models.
        https://github.com/salesforce/Merlion
        """
        from merlion.models.factory import ModelFactory
        from merlion.transform.resample import TemporalResample
        from merlion.utils import TimeSeries

        def run_model(config):
            print(f"Running: {config}")

            model_kwargs = config["model_kwargs"]
            model_kwargs["transform"] = TemporalResample(
                granularity=None,
                aggregation_policy="Mean",
                missing_value_policy="FFill",
            )
            model_kwargs["target_seq_index"] = self.target_index

            model = ModelFactory.create(
                name=config["model_name"],
                max_forecast_steps=self.forecast_steps,
                **model_kwargs,
            )

            train_ts = TimeSeries.from_pd(self.train_df)
            model.train(train_ts)

            forecast, err = model.forecast(self.forecast_steps)
            forecast = forecast.to_pd().values
            return dict(id=config["id"], forecast=forecast)

        self.forecasts = parallel_map(
            run_model, self.model_config["libs"].get("merlion", [])
        )
        self.next(self.join)

    # We must use pip instead of conda because mxnet 1.5.0 is broken.
    # @batch(cpu=2, memory=4000)
    @pip(libraries={"mxnet": "1.8.0.post0", "gluonts": "0.8.1"})
    @conda(libraries={"ipykernel": IPYKERNEL_VERSION, "pip": PIP_VERSION})
    @step
    def backtest_gluonts(self):
        """
        Backtest gluon-ts models.
        https://github.com/awslabs/gluon-ts
        """
        import numpy as np
        import pandas as pd
        from pydoc import locate
        from gluonts.dataset.common import ListDataset
        from gluonts.model.forecast import SampleForecast
        from gluonts.mx.trainer import Trainer

        def df_to_dset(df):
            return ListDataset(
                [
                    {
                        "start": df.index[0],
                        "target": df.values[:, self.target_index],
                    },
                ],
                freq=self.freq,
            )

        def run_model(config):
            print(f"Running: {config}")

            EstimatorClass = locate(config["estimator_class"])

            trainer_kwargs = config.get("trainer_kwargs", None)
            if trainer_kwargs is not None:
                # trainer_kwargs["epochs"] = 2
                trainer = Trainer(**trainer_kwargs)
            else:
                trainer = None

            if trainer is not None:
                estimator = EstimatorClass(
                    prediction_length=self.forecast_steps,
                    freq=self.freq,
                    trainer=trainer,
                    **config.get("estimator_kwargs", {}),
                )
            else:
                estimator = EstimatorClass(
                    prediction_length=self.forecast_steps,
                    freq=self.freq,
                    **config.get("estimator_kwargs", {}),
                )

            train_dset = df_to_dset(self.train_df)
            predictor = estimator.train(training_data=train_dset)

            # An iterator over the number of targets (only one for univariate)
            forecast = predictor.predict(train_dset, num_samples=100)

            # For models like DeepAR, will be (num_samples x self.forecast_steps),
            # for models without sampling, will be (1 x self.forecast_steps).
            forecast = next(forecast)

            # some models return QuantileForecast instead
            assert isinstance(forecast, SampleForecast)

            # take mean over samples
            forecast = np.mean(forecast.samples, 0)

            return dict(id=config["id"], forecast=forecast)

        self.forecasts = parallel_map(
            run_model, self.model_config["libs"].get("gluonts", [])
        )
        self.next(self.join)

    # @batch(cpu=2, memory=4000)
    @conda(libraries={"ipykernel": IPYKERNEL_VERSION, "kats": "0.1.0"})
    @step
    def backtest_kats(self):
        """
        Backtest Kats models.
        https://github.com/facebookresearch/Kats
        """

        from pydoc import locate
        from kats.consts import TimeSeriesData

        def df_to_data(df):
            df2 = df.iloc[:, [self.target_index]].reset_index()
            df2.columns = ["time", "value"]
            return TimeSeriesData(df2)

        def run_model(config):
            print(f"Running: {config}")

            ModelClass = locate(config["model_class"])
            ParamsClass = locate(config["params_class"])

            params = ParamsClass(**config.get("params_kwargs", {}))
            train_data = df_to_data(self.train_df)
            model = ModelClass(train_data, params)
            model.fit()

            # some models have columns "fcst_lower" and "fcst_upper" for upper and lower bounds
            forecast = model.predict(steps=self.forecast_steps, freq=self.freq)
            forecast = forecast["fcst"].values

            return dict(id=config["id"], forecast=forecast)

        self.forecasts = parallel_map(
            run_model, self.model_config["libs"].get("kats", [])
        )
        self.next(self.join)

    # @batch(cpu=2, memory=4000)
    @pip(libraries={"neuralprophet": "0.3.0"})
    @conda(libraries={"ipykernel": IPYKERNEL_VERSION, "pip": PIP_VERSION})
    @step
    def backtest_neuralprophet(self):
        """
        Backtest NeuralProphet models.
        https://github.com/ourownstory/neural_prophet
        """

        from pydoc import locate
        from neuralprophet import NeuralProphet

        def convert_df(df):
            # requires a column named "ds" for dates and "y" for target
            df2 = df.iloc[:, [self.target_index]].reset_index()
            df2.columns = ["ds", "y"]
            return df2

        def run_model(config):
            print(f"Running: {config}")

            train_df = convert_df(self.train_df)

            model = NeuralProphet(**config.get("model_kwargs", {}))
            model.fit(train_df, freq=self.freq)

            future = model.make_future_dataframe(train_df, periods=self.forecast_steps)
            forecast = model.predict(future)
            forecast = forecast["yhat1"].values

            return dict(id=config["id"], forecast=forecast)

        self.forecasts = parallel_map(
            run_model, self.model_config["libs"].get("neuralprophet", [])
        )
        self.next(self.join)

    @conda(libraries={"ipykernel": IPYKERNEL_VERSION, "pandas": PANDAS_VERSION})
    @step
    def join(self, inputs):
        """
        Compute performance metrics for each library.
        """
        from collections import OrderedDict
        import numpy as np
        import pandas as pd

        forecasts = OrderedDict()
        for lib in inputs:
            self.test_df = lib.test_df
            self.target_index = lib.target_index
            for forecast in lib.forecasts:
                assert (
                    forecast["id"] not in forecasts
                ), f"Duplicate forecast id: {forecast['id']}"
                forecasts[forecast["id"]] = forecast["forecast"].reshape(-1)

        self.forecasts = pd.DataFrame(forecasts)
        print(self.forecasts)

        if self.test_df is not None:
            # duplicate univariate target across columns for each model
            true = self.test_df.iloc[
                : self.forecast_steps, [self.target_index] * self.forecasts.shape[1]
            ]
            pred = self.forecasts
            print(true)
            print("--")
            print(pred)
            rmse = np.sqrt(np.mean((pred.values - true.values) ** 2, 1))
            step_rmse = pd.DataFrame(np.sqrt((pred.values - true.values) ** 2))
            print(f"RMSE: {rmse}")
            print(f"Step RMSE:")
            print(step_rmse)

        #         error = np.zeros((inp.forecast_steps, 1))
        #         for window in model_windows:
        #             # univariate scoring
        #             true = window["true"][:, inp.target_index].reshape(-1, 1)
        #             pred = window["pred"].reshape(-1, 1)
        #             error += (pred - true) ** 2
        #             model_id = window["id"]  # all have same id

        #         rmse = np.sqrt(np.mean(error))
        #         errors[model_id] = error
        #         rmses[model_id] = rmse
        #         # TODO: handle multiple retrainings

        # print(pd.Series(rmses, name="RMSE").sort_values())

        # self.errors = errors
        # self.rmses = rmses

        self.next(self.end)

    @step
    def end(self):
        """
        End of the flow
        """
        pass


if __name__ == "__main__":
    ForecastingFlow()
