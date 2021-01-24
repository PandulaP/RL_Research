#!/bin/bash

pip install -e ../../../environments/custom_cartpole/.

python experiment_runs/experiment_left_skewed_init_dist.py &
python experiment_runs/experiment_right_skewed_init_dist.py &
python experiment_runs/parameter_config_unbiased_init_dist.py