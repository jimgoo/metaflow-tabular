"""
A set of wrapper classes around forecasting libraries. The imports are inside
the method calls so that the libraries only have to be installed inside the
the Metaflow step environments.
"""
from pydoc import locate


class BaseModel:
    def __init__(self, target_index, steps_ahead, data_freq=None):
        self.target_index = target_index
        self.steps_ahead = steps_ahead
        self.data_freq = data_freq

    def fit(self, train_df):
        """
        Fit the model to the training dataframe.
        """
        pass

    def predict(self, context_df=None):
        """
        Predict future values using the context dataframe.
        """
        pass

    def df_to_data_type(self, df):
        """
        Convert a dataframe to the data type of the model.
        """
        return df


class MerlionModel(BaseModel):
    def __init__(self, config, target_index, steps_ahead, data_freq=None):
        from merlion.models.factory import ModelFactory
        from merlion.transform.resample import TemporalResample

        super().__init__(target_index, steps_ahead, data_freq)

        model_kwargs = config["model_kwargs"]
        model_kwargs["transform"] = TemporalResample(
            granularity=None,
            aggregation_policy="Mean",
            missing_value_policy="FFill",
        )
        model_kwargs["target_seq_index"] = self.target_index

        self.model = ModelFactory.create(
            name=config["model_name"],
            max_forecast_steps=self.steps_ahead,
            **model_kwargs,
        )

    def fit(self, train_df):
        train_ts = self.df_to_data_type(train_df)
        self.model.train(train_ts)

    def predict(self, context_df=None):
        time_series_prev = self.df_to_data_type(context_df)

        # build time_stamps instead of using steps_ahead since an integer only works with some models
        freq = context_df.index[1] - context_df.index[0]
        time_stamps = [
            context_df.index[-1] + (i + 1) * freq for i in range(self.steps_ahead)
        ]

        y_hat, err = self.model.forecast(time_stamps, time_series_prev=time_series_prev)
        y_hat = y_hat.to_pd()
        err = err.to_pd() if err is not None else None

        return dict(y_hat=y_hat.values, y_dates=y_hat.index, err=err)

    def df_to_data_type(self, df):
        from merlion.utils import TimeSeries

        return TimeSeries.from_pd(df)


class GluonTSModel(BaseModel):
    def __init__(self, config, target_index, steps_ahead, data_freq=None):
        from gluonts.mx.trainer import Trainer

        super().__init__(target_index, steps_ahead, data_freq)

        EstimatorClass = locate(config["estimator_class"])

        trainer_kwargs = config.get("trainer_kwargs", None)
        if trainer_kwargs is not None:
            trainer = Trainer(**trainer_kwargs)
        else:
            trainer = None

        if trainer is not None:
            self.estimator = EstimatorClass(
                prediction_length=self.steps_ahead,
                freq=self.data_freq,
                trainer=trainer,
                **config.get("estimator_kwargs", {}),
            )
        else:
            self.estimator = EstimatorClass(
                prediction_length=self.steps_ahead,
                freq=self.data_freq,
                **config.get("estimator_kwargs", {}),
            )

    def fit(self, train_df):
        training_data = self.df_to_data_type(train_df)
        self.predictor = self.estimator.train(training_data=training_data)

    def predict(self, context_df=None):
        import numpy as np
        from gluonts.model.forecast import SampleForecast

        context_data = self.df_to_data_type(context_df)

        # An iterator over the number of targets (only one for univariate)
        forecast = self.predictor.predict(context_data, num_samples=100)

        # For models like DeepAR, will be (num_samples x self.forecast_steps),
        # for models without sampling, will be (1 x self.forecast_steps).
        forecast = next(forecast)

        # some models return QuantileForecast instead
        assert isinstance(forecast, SampleForecast), "QuantileForecast not supported"

        # take mean over samples
        y_hat = np.mean(forecast.samples, 0)

        return dict(y_hat=y_hat, samples=forecast.samples)

    def df_to_data_type(self, df):
        from gluonts.dataset.common import ListDataset

        return ListDataset(
            [
                {
                    "start": df.index[0],
                    "target": df.values[:, self.target_index],
                },
            ],
            freq=self.data_freq,
        )


class KatsModel(BaseModel):
    def __init__(self, config, target_index, steps_ahead, data_freq=None):
        super().__init__(target_index, steps_ahead, data_freq)

        params_class = locate(config["params_class"])
        self.params = params_class(**config.get("params_kwargs", {}))
        self.model_class = locate(config["model_class"])

    def fit(self, train_df):
        train_data = self.df_to_data_type(train_df)
        self.model = self.model_class(train_data, self.params)
        self.model.fit()

    def predict(self, context_df=None):
        forecast = self.model.predict(steps=self.steps_ahead, freq=self.data_freq)

        # some models have "fcst_lower" and "fcst_upper" columns
        y_hat = forecast["fcst"].values
        y_dates = forecast["time"].values
        return dict(y_hat=y_hat, y_dates=y_dates)

    def df_to_data_type(self, df):
        from kats.consts import TimeSeriesData

        df2 = df.iloc[:, [self.target_index]].reset_index()
        df2.columns = ["time", "value"]
        return TimeSeriesData(df2)


class NeuralProphetModel(BaseModel):
    def __init__(self, config, target_index, steps_ahead, data_freq=None):
        from neuralprophet import NeuralProphet

        super().__init__(target_index, steps_ahead, data_freq)

        self.model = NeuralProphet(**config.get("model_kwargs", {}))

    def fit(self, train_df):
        train_df = self.df_to_data_type(train_df)
        self.model.fit(train_df, freq=self.data_freq)

    def predict(self, context_df=None):
        context_df = self.df_to_data_type(context_df)
        future = self.model.make_future_dataframe(context_df, periods=self.steps_ahead)
        forecast = self.model.predict(future)

        y_hat = forecast["yhat1"].values
        y_dates = forecast["ds"].values

        return dict(y_hat=y_hat, y_dates=y_dates)

    def df_to_data_type(self, df):
        """
        NeuralProphet requires a column named "ds" for dates and "y" for target.
        """
        df2 = df.iloc[:, [self.target_index]].reset_index()
        df2.columns = ["ds", "y"]
        return df2
