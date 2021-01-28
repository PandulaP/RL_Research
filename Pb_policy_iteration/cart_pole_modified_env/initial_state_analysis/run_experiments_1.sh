#!/bin/bash

# running experiments with the original algorithm
pip install -e ../../../environments/custom_cartpole/. &&
python experiment_runs/experiment_left_skewed_init_dist_org_algo.py &
python experiment_runs/experiment_right_skewed_init_dist_org_algo.py &
python experiment_runs/parameter_config_unbiased_init_dist_org_algo.py