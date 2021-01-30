#!/bin/bash

# running experiments with the modified algorithm
pip install -e ../../../../environments/custom_cartpole/. &&
python parameter_config_1.py &
python parameter_config_2.py &
python parameter_config_3.py &
python parameter_config_4.py &
python parameter_config_5.py &
python parameter_config_6.py &
python parameter_config_7.py &
python parameter_config_8.py &
python parameter_config_9.py &
python parameter_config_10.py &
python parameter_config_11.py
