# First, install the custom cart-pole environment
pip install -e ../pbpi/custom_cartpole/. &&

# Then, run the python experiments
python exp_modified_config_2.py &
python exp_orginal_config_2.py