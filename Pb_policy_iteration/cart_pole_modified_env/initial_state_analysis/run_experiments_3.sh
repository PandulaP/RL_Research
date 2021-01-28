#!/bin/bash

pip install -e ../../../environments/custom_cartpole/. &&
python experiment_runs/parameter_config_unbiased_init_dist_small_org_algo.py &
python experiment_runs/parameter_config_unbiased_init_dist_large_org_algo.py