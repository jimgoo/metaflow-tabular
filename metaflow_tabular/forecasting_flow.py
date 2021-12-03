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

PIP_VERSION = "21.3.1"


def prediction_iterator(n_train, n_test, forecast_steps):
    """
    Generate a sequence of intervals for making forecasts.

    `pred_interval` is a tuple of the form (start, end),
    where start and end are integers that specify the range of
    past time-series values to use for making a forecast.

    `true_interval` is a tuple of the form (start, end),
    where start and end are integers that specify the range of
    true future time-series values to use for scoring.

    :param int n_train: number of training samples
    :param int n_test: number of test samples
    :param int forecast_steps: number of steps ahead to forecast
    :return: iterator of tuples of the form (pred_interval, true_interval)
    """
    i = 0
    start, end = 0, n_train

    while end < n_train + n_test - forecast_steps:
        end = n_train + i * forecast_steps
        pred_interval = (start, end)
        true_interval = (end, end + forecast_steps)
        i += 1
        yield pred_interval, true_interval


@conda_base(python="3.8.12")
class ForecastingFlow(FlowSpec):
    """
    A flow for benchmarking forecasting libraries.
    """

    df_path = Parameter(
        "df_path",
        help="The path to a DataFrame CSV file",
        default="https://jgoode.s3.amazonaws.com/ts-datasets/seattle-trail.csv",
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

        df = pd.read_csv(self.df_path)

        assert self.date_col in df.columns, '"%s" not in columns' % self.date_col
        assert self.target_col in df.columns, '"%s" not in columns' % self.target_col

        # parse date column and set it as the index
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df.set_index(self.date_col, inplace=True)

        # get index of the target column
        self.target_index = df.columns.tolist().index(self.target_col)

        # get the frequency of the data
        self.freq = pd.infer_freq(df.index)

        # this will pickle the DataFrame for other steps to use
        self.df = df

        # load the model config file
        with open(self.model_config_path, "r") as f:
            self.model_config = yaml.safe_load(f)

        # setup the backtesting windows
        self.n_train = 500
        self.n_test = 100
        self.forecast_steps = 10

        self.backtest_windows = [
            dict(
                train_start=0,
                train_end=self.n_train,
                pred_intervals=list(
                    prediction_iterator(self.n_train, self.n_test, self.forecast_steps)
                ),
            )
        ]

        print("DataFrame")
        print(self.df)
        print("model_config")
        pprint(self.model_config)
        print("backtest_windows")
        print(self.backtest_windows)

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

            forecasts = []
            for iw, window in enumerate(self.backtest_windows):
                train_ts = TimeSeries.from_pd(
                    self.df.iloc[window["train_start"] : window["train_end"]]
                )

                model.reset()
                model.train(train_ts)

                for ip, (pred_interval, true_inverval) in enumerate(
                    window["pred_intervals"]
                ):
                    prev = self.df.iloc[pred_interval[0] : pred_interval[1]]
                    true = self.df.iloc[true_inverval[0] : true_inverval[1]]

                    forecast, err = model.forecast(
                        true.index, time_series_prev=TimeSeries.from_pd(prev)
                    )
                    forecast = forecast.to_pd().values
                    # err = err.to_pd() if err is not None else None

                    forecasts.append(
                        dict(
                            id=config["id"],
                            iw=iw,
                            ip=ip,
                            true=true.values,
                            pred=forecast,
                        )
                    )

            return forecasts

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

        def run_model(config):
            print(f"Running: {config}")

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

            forecasts = []
            for iw, window in enumerate(self.backtest_windows):
                train_dset = df_to_dset(
                    self.df.iloc[window["train_start"] : window["train_end"]]
                )

                predictor = estimator.train(training_data=train_dset)

                for ip, (pred_interval, true_inverval) in enumerate(
                    window["pred_intervals"]
                ):
                    prev = self.df.iloc[pred_interval[0] : pred_interval[1]]
                    true = self.df.iloc[true_inverval[0] : true_inverval[1]]

                    prev_dset = df_to_dset(prev)

                    # An iterator over the number of targets (only one for univariate)
                    forecast = predictor.predict(prev_dset, num_samples=100)

                    # For models like DeepAR, will be (num_samples x self.forecast_steps),
                    # for models without sampling, will be (1 x self.forecast_steps).
                    forecast = next(forecast)

                    # some models return QuantileForecast instead
                    assert isinstance(forecast, SampleForecast)

                    # take mean over samples
                    forecast = np.mean(forecast.samples, 0)

                    forecasts.append(
                        dict(
                            id=config["id"],
                            iw=iw,
                            ip=ip,
                            true=true.values,
                            pred=forecast,
                        )
                    )

            return forecasts

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

        def run_model(config):
            print(f"Running: {config}")

            def df_to_data(df):
                df2 = df.iloc[:, [self.target_index]].reset_index()
                df2.columns = ["time", "value"]
                return TimeSeriesData(df2)

            ModelClass = locate(config["model_class"])
            ParamsClass = locate(config["params_class"])

            params = ParamsClass(**config.get("params_kwargs", {}))

            forecasts = []
            for iw, window in enumerate(self.backtest_windows):
                train_data = df_to_data(
                    self.df.iloc[window["train_start"] : window["train_end"]]
                )

                model = ModelClass(train_data, params)
                model.fit()

                for ip, (pred_interval, true_inverval) in enumerate(
                    window["pred_intervals"]
                ):
                    # prev = self.df.iloc[pred_interval[0] : pred_interval[1]]
                    true = self.df.iloc[true_inverval[0] : true_inverval[1]]

                    # TODO: allow for using prev data
                    forecast = model.predict(steps=self.forecast_steps, freq=self.freq)
                    # some models have columns "fcst_lower" and "fcst_upper" for upper and lower bounds
                    forecast = forecast["fcst"].values

                    forecasts.append(
                        dict(
                            id=config["id"],
                            iw=iw,
                            ip=ip,
                            true=true.values,
                            pred=forecast,
                        )
                    )

            return forecasts

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

        def run_model(config):
            print(f"Running: {config}")

            def convert_df(df):
                # requires a column named "ds" for dates and "y" for target
                df2 = df.iloc[:, [self.target_index]].reset_index()
                df2.columns = ["ds", "y"]
                return df2

            forecasts = []
            for iw, window in enumerate(self.backtest_windows):
                train_df = self.df.iloc[window["train_start"] : window["train_end"]]
                train_df = convert_df(train_df)

                model = NeuralProphet(**config.get("model_kwargs", {}))
                model.fit(train_df, freq=self.freq)

                for ip, (pred_interval, true_inverval) in enumerate(
                    window["pred_intervals"]
                ):
                    prev = self.df.iloc[pred_interval[0] : pred_interval[1]]
                    true = self.df.iloc[true_inverval[0] : true_inverval[1]]

                    prev = convert_df(prev)

                    future = model.make_future_dataframe(
                        prev, periods=self.forecast_steps
                    )
                    print("---")
                    print(future)
                    forecast = model.predict(future)
                    forecast = forecast["yhat1"].values

                    forecasts.append(
                        dict(
                            id=config["id"],
                            iw=iw,
                            ip=ip,
                            true=true.values,
                            pred=forecast,
                        )
                    )

            return forecasts

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

        errors = OrderedDict()
        rmses = OrderedDict()
        for inp in inputs:
            for model_windows in inp.forecasts:
                error = np.zeros((inp.forecast_steps, 1))
                for window in model_windows:
                    # univariate scoring
                    true = window["true"][:, inp.target_index].reshape(-1, 1)
                    pred = window["pred"].reshape(-1, 1)
                    error += (pred - true) ** 2
                    model_id = window["id"]  # all have same id

                rmse = np.sqrt(np.mean(error))
                errors[model_id] = error
                rmses[model_id] = rmse
                # TODO: handle multiple retrainings

        print(pd.Series(rmses, name="RMSE").sort_values())

        self.errors = errors
        self.rmses = rmses

        self.next(self.end)

    @step
    def end(self):
        """
        Cleanup and exit.
        """
        pass


if __name__ == "__main__":
    ForecastingFlow()
