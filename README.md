# metaflow-tabular

[![pypi](https://img.shields.io/pypi/v/metaflow-tabular.svg)](https://pypi.org/project/metaflow-tabular)
[![tests](https://github.com/jimgoo/metaflow-tabular/actions/workflows/tests.yml/badge.svg)](https://github.com/jimgoo/metaflow-tabular/actions)
[![docs](https://github.com/jimgoo/metaflow-tabular/actions/workflows/docs.yml/badge.svg)](https://jimgoo.github.io/metaflow-tabular)

Lots of tabular models running in Metaflow.

## Installation

```bash
pip install metaflow-tabular
```

The only dependency is [Metaflow](https://github.com/Netflix/metaflow/).  We make use of its bulit-in conda decorator where possible and use the included pip decorator otherwise. Therefore you will need the conda package manager to be installed with the conda-forge channel added.
   1. Download Miniconda at https://docs.conda.io/en/latest/miniconda.html
   2. ```conda config --add channels conda-forge```

For dramatically faster conda environment operations, you can install [mamba](https://github.com/mamba-org/mamba) and use it with [this fork of Metaflow](https://github.com/jimgoo/metaflow) that's in code review.

## Contributing

We welcome contributions, please see our [contribution guide](https://jimgoo.github.io/metaflow-tabular/contributing.html) for more details.
