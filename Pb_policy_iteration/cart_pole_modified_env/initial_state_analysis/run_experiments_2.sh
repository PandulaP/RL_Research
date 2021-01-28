#!/bin/bash

pip install -e ../../../environments/custom_cartpole/. &&
python experiment_runs/parameter_config_unbiased_init_dist_small_mod_algo.py &
python experiment_runs/parameter_config_unbiased_init_dist_large_mod_algo.py