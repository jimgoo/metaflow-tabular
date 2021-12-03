#!/usr/bin/env python
import os


def test_run():
    exit_code = os.system(
        "cd metaflow_tabular && \
            conda config --add channels conda-forge && \
            python forecasting_flow.py --environment=conda run"
    )
    print("exit_code", exit_code)
    assert exit_code == 0, exit_code
