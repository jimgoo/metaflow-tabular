"""
This flow requires the 'conda' package manager to be installed with the
conda-forge channel added.
   a. Download Miniconda at https://docs.conda.io/en/latest/miniconda.html
   b. ```conda config --add channels conda-forge```

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

# install this for all steps (helpful with jupyter)
IPYKERNEL_VERSION = "6.5.0"


def prediction_iterator(n_train, n_test, forecast_steps):
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

    model_config = Parameter(
        "model_config",
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
        print("Using metadata provider: %s" % get_metadata())

        df = pd.read_csv(self.df_path)

        assert self.date_col in df.columns, '"%s" not in columns' % self.date_col
        assert self.target_col in df.columns, '"%s" not in columns' % self.target_col

        # parse date column and set it as the index
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df.set_index(self.date_col, inplace=True)

        # get index of the target column
        self.target_index = df.columns.tolist().index(self.target_col)

        # this will pickle the DataFrame for other steps to use
        self.df = df

        # load the model config file
        with open(self.model_config, "r") as f:
            self.models = yaml.safe_load(f)

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

        print("Input DataFrame")
        print(self.df)
        print("Models")
        pprint(self.models)
        print("backtest_windows")
        print(self.backtest_windows)

        # branches will run in parallel
        self.next(self.fit_merlion, self.fit_gluonts)

    # @batch(cpu=2, memory=4000)
    @conda(libraries={"ipykernel": IPYKERNEL_VERSION, "salesforce-merlion": "1.0.2"})
    @step
    def fit_merlion(self):
        """
        Fit the Merlion models.
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

        self.forecasts = parallel_map(run_model, self.models["merlion"])
        self.next(self.join)

    # @batch(cpu=2, memory=4000)
    @pip(libraries={"mxnet": "1.8.0.post0", "gluonts": "0.8.1"})
    @conda(libraries={"ipykernel": IPYKERNEL_VERSION, "pip": "21.3.1"})
    @step
    def fit_gluonts(self):
        """
        Fit gluon-ts models.
        We must use pip instead of conda because mxnet 1.5.0 is broken.
        """
        import numpy as np
        import pandas as pd
        from pydoc import locate
        from gluonts.dataset.common import ListDataset
        from gluonts.model.forecast import SampleForecast
        from gluonts.mx.trainer import Trainer

        df_freq = pd.infer_freq(self.df.index)
        print("df_freq", df_freq)

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
                    freq=df_freq,
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
                    freq=df_freq,
                    trainer=trainer,
                    **config.get("estimator_kwargs", {}),
                )
            else:
                estimator = EstimatorClass(
                    prediction_length=self.forecast_steps,
                    freq=df_freq,
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

        self.forecasts = parallel_map(run_model, self.models["gluonts"])
        self.next(self.join)

    @conda(libraries={"ipykernel": IPYKERNEL_VERSION, "pandas": PANDAS_VERSION})
    @step
    def join(self, inputs):
        """
        Join our parallel branches and merge results.
        """
        from collections import OrderedDict
        import numpy as np
        import pandas as pd

        errors = OrderedDict()
        for inp in inputs:
            for model_windows in inp.forecasts:
                error = np.zeros((inp.forecast_steps, 1))
                for window in model_windows:
                    # univariate scoring
                    true = window["true"][:, inp.target_index].reshape(-1, 1)
                    pred = window["pred"].reshape(-1, 1)
                    error += (pred - true) ** 2

                rmse = np.sqrt(np.mean(error))
                model_id = window["id"]
                errors[model_id] = rmse

        print(pd.Series(errors).sort_values())

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    ForecastingFlow()
