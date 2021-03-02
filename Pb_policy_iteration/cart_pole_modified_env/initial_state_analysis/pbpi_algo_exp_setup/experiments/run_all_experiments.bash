# First, install the custom cart-pole environment
pip install -e ../pbpi/custom_cartpole/. &&

# Then, run the python experiments
python exp_modified_config_1.py &
python exp_modified_config_2.py &
python exp_modified_config_3.py &
python exp_modified_config_4.py &
python exp_original_config_1.py &
python exp_original_config_2.py &
python exp_original_config_3.py &
python exp_original_config_4.py 