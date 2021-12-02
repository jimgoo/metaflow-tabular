#!/usr/bin/env python
import os


def test_run():
    exit_code = os.system(
        "cd metaflow_tabular && python forecasting_flow.py --environment=conda run --model_config ../configs/forecasting/models/test.yaml"
    )
    print("exit_code", exit_code)
    assert exit_code == 0, exit_code
