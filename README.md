# metaflow-tabular

[![pypi](https://img.shields.io/pypi/v/metaflow-tabular.svg)](https://pypi.org/project/metaflow-tabular)
[![tests](https://github.com/jimgoo/metaflow-tabular/actions/workflows/tests.yml/badge.svg)](https://github.com/jimgoo/metaflow-tabular/actions)
[![docs](https://github.com/jimgoo/metaflow-tabular/actions/workflows/docs.yml/badge.svg)](https://jimgoo.github.io/metaflow-tabular)

Tabular models running in Metaflow.

## Installation

```bash
pip install metaflow-tabular
```

The only dependency is [Metaflow](https://github.com/Netflix/metaflow/).  We make use of its bulit-in conda decorator where possible and use the included pip decorator otherwise. Therefore you will need the conda package manager to be installed with the conda-forge channel added.
   1. Download Miniconda at https://docs.conda.io/en/latest/miniconda.html
   2. ```conda config --add channels conda-forge```

For faster conda operations, install [mamba](https://github.com/mamba-org/mamba) and use it with [this fork of Metaflow](https://github.com/jimgoo/metaflow) that's in code review.

## Usage

The forecasting flow backtests several models from Salesforce's Merlion, Amazons's GluonTS, and Facebook's Kats libraries in parallel using branched steps:

```bash
python forecasting_flow.py --environment=conda show
```

```
A flow for benchmarking forecasting libraries.

Step start
    Start the flow by preprocessing the data.
    => fit_merlion, fit_gluonts

Step fit_merlion
    Fit the Merlion models.
    => join

Step fit_gluonts
    Fit gluon-ts models.
    => join

Step join
    Compute performance metrics for each library.
    => end

Step end
    Cleanup and exit.
```

You can run it via the `run` command:

```bash
python forecasting_flow.py --environment=conda run --help
```

```
Usage: forecasting_flow.py run [OPTIONS]

  Run the workflow locally.

Options:
  --df_path TEXT            The path to a DataFrame CSV file  [default:
                            https://jgoode.s3.amazonaws.com/ts-
                            datasets/seattle-trail.csv]
  --date_col TEXT           Column of the date in the input DataFrame
                            [default: Date]
  --target_col TEXT         Column of the target in the input DataFrame
                            [default: BGT North of NE 70th Total]
  --model_config TEXT       The path to a model config file  [default:
                            ../configs/forecasting/models/default.yaml]                            
```

The `default.yaml` model config describes which models from each library to run:

```yaml
---

merlion:
  - id: "merlion-Arima1"
    model_name: "Arima"
    model_kwargs:
      order: [30, 0, 10]
      
  - id: "merlion-Arima2"
    model_name: "Arima"
    model_kwargs:
      order: [10, 0, 5]

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
``` 

## Contributing

Contributions are welcome, please see the [contribution guide](https://jimgoo.github.io/metaflow-tabular/contributing.html) for more details.
