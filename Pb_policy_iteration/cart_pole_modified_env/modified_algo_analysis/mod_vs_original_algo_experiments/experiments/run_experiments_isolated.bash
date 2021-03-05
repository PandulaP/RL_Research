# First, install the custom cart-pole environment
pip install -e ../pbpi/custom_cartpole/. &&

# Then, run the python experiments
python exp_modified_config_10.py &
python exp_modified_config_11.py &
python exp_modified_config_12.py &
python exp_original_config_10.py &
python exp_original_config_11.py &
python exp_original_config_12.py