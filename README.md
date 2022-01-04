# metaflow-tabular

[![pypi](https://img.shields.io/pypi/v/metaflow-tabular.svg)](https://pypi.org/project/metaflow-tabular)
[![tests](https://github.com/jimgoo/metaflow-tabular/actions/workflows/tests.yml/badge.svg)](https://github.com/jimgoo/metaflow-tabular/actions)
[![docs](https://github.com/jimgoo/metaflow-tabular/actions/workflows/docs.yml/badge.svg)](https://jimgoo.github.io/metaflow-tabular)

Tabular models from popular open-source libraries running in Metaflow. Supports:

- [Merlion](https://github.com/salesforce/Merlion)
- [GluonTS](https://github.com/awslabs/gluon-ts)
- [Kats](https://github.com/facebookresearch/Kats)
- [NeuralProphet](https://github.com/ourownstory/neural_prophet)

## Installation

To install, run

```bash
pip install metaflow-tabular
```

The only dependency is [Metaflow](https://github.com/Netflix/metaflow/).  We make use of its bulit-in conda decorator where possible and use the included pip decorator otherwise. Therefore you will need the conda package manager to be installed with the conda-forge channel added.
   1. Download Miniconda at https://docs.conda.io/en/latest/miniconda.html
   2. ```conda config --add channels conda-forge```

For faster conda operations, install [mamba](https://github.com/mamba-org/mamba) and use it with [this fork of Metaflow](https://github.com/jimgoo/metaflow) that's under review.

## Usage

The forecasting flow fits models and makes forecasts using several popular time-series libraries. Each library is a branch in the DAG that can be run in parallel.

```bash
python forecasting_flow.py --environment=conda show
```

```
A flow for benchmarking forecasting libraries.

Step start
    Start the flow by preprocessing the data.
    => run_merlion, run_gluonts, run_kats, run_neuralprophet

Step run_merlion
    Backtest Merlion models.
    https://github.com/salesforce/Merlion
    => join

Step run_gluonts
    Backtest gluon-ts models.
    https://github.com/awslabs/gluon-ts
    => join

Step run_kats
    Backtest Kats models.
    https://github.com/facebookresearch/Kats
    => join

Step run_neuralprophet
    Backtest NeuralProphet models.
    https://github.com/ourownstory/neural_prophet
    => join

Step join
    Compute performance metrics for each library.
    => end

Step end
    End of the flow
```

You can run it via the `run` command:

```bash
python forecasting_flow.py --environment=conda run --help
```

```
Usage: forecasting_flow.py run [OPTIONS]

  Run the workflow locally.

Options:
  --train_path TEXT         The path to a DataFrame file for training
                            [default: https://jgoode.s3.amazonaws.com/ts-
                            datasets/seattle-trail.csv]
  --test_path TEXT          The path to a DataFrame file for testing
  --date_col TEXT           Column of the date in the input DataFrame
                            [default: Date]
  --target_col TEXT         Column of the target in the input DataFrame
                            [default: BGT North of NE 70th Total]
  --model_config_path TEXT  The path to a model config file  [default:
                            ../configs/forecasting/models/default.yaml]
  --forecast_steps INTEGER  The number of steps ahead to forecast  [default:
                            10]
```

The `default.yaml` model config includes all models supported from each library:


```yaml
---

libs:
  merlion:
    - id: "merlion-Arima"
      model_name: "Arima"
      model_class: "merlion.models.forecast.arima.Arima"
      model_kwargs: {}

    - id: "merlion-Sarima"
      model_name: "Sarima"
      model_class: "merlion.models.forecast.sarima.Sarima"
      model_kwargs: {}

    - id: "merlion-ets"
      model_name: "ETS"
      model_class: "merlion.models.forecast.ets.ETS"
      model_kwargs: {}

    - id: "merlion-MSES"
      model_name: "MSES"
      model_class: "merlion.models.forecast.smoother.MSES"
      model_kwargs: {}

    - id: "merlion-VAR"
      model_name: "VectorAR"
      model_class: "merlion.models.forecast.vector_ar.VectorAR"
      model_kwargs:
        # required
        maxlags: 10

    - id: "merlion-RandomForest"
      model_name: "RandomForestForecaster"
      model_class: "merlion.models.forecast.baggingtrees.RandomForestForecaster"
      model_kwargs:
        # required
        maxlags: 10

    - id: "merlion-ExtraTrees"
      model_name: "ExtraTreesForecaster"
      model_class: "merlion.models.forecast.baggingtrees.ExtraTreesForecaster"
      model_kwargs:
        # required
        maxlags: 10

    - id: "merlion-LGBM"
      model_name: "LGBMForecaster"
      model_class: "merlion.models.forecast.boostingtrees.LGBMForecaster"
      model_kwargs:
        # required
        maxlags: 10

  gluonts:
    - id: "gluonts-SeasonalNaive"
      estimator_class: "gluonts.model.seasonal_naive.SeasonalNaiveEstimator"
      estimator_kwargs: {}
      # no trainer

    - id: "gluonts-NPTS"
      estimator_class: "gluonts.model.npts.NPTSEstimator"
      estimator_kwargs: {}
      # no trainer

    - id: "gluonts-DeepAR"
      estimator_class: "gluonts.model.deepar.DeepAREstimator"
      estimator_kwargs: {}
      trainer_kwargs: {}

    - id: "gluonts-SimpleFeedForward"
      estimator_class: "gluonts.model.simple_feedforward.SimpleFeedForwardEstimator"
      estimator_kwargs: {}
      trainer_kwargs: {}

    - id: "gluonts-Transformer"
      estimator_class: "gluonts.model.transformer.TransformerEstimator"
      estimator_kwargs: {}
      trainer_kwargs: {}

  kats:
    - id: "kats-prophet"
      model_class: "kats.models.prophet.ProphetModel"
      params_class: "kats.models.prophet.ProphetParams"
      params_kwargs: {}

    - id: "kats-holtwinters"
      model_class: "kats.models.holtwinters.HoltWintersModel"
      params_class: "kats.models.holtwinters.HoltWintersParams"
      params_kwargs: {}

  neuralprophet:
    - id: "neuralprophet-default"
      model_class: "neuralprophet.NeuralProphet"
      model_kwargs: {}
```

The above uses the deafult parameters for each model, but you can define your own config file with custom parameters and provide it to the flow via `--model_config_path`. For example, to limit the number of epochs for a GluonTS DeepAR model:

```yaml
  gluonts:
    - id: "gluonts-DeepAR"
      estimator_class: "gluonts.model.deepar.DeepAREstimator"
      estimator_kwargs: {}
      trainer_kwargs:
        epochs: 30
```

## Contributing

Contributions are welcome, please see the [contribution guide](https://jimgoo.github.io/metaflow-tabular/contributing.html) for more details.
