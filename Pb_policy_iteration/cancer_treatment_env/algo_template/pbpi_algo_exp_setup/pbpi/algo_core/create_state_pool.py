import gym
from gym import wrappers
import custom_cartpole  # custom cart-pole environment

import pbpi.algo_core.algo_file_paths as f_paths

import numpy as np
import pandas as pd

from pathlib import Path

###################################################

def create_initial_state_set(sample_size:int):

    # Initial patient set
    tumor_size_init, toxicity_init = np.random.uniform(low=0, high=2, size = (2, num_samples))
    init_patients = [(t_size, toxicity) for t_size, toxicity in zip(tumor_size_init,toxicity_init)]
    
    return init_patients


if __name__ == '__main__':
    sample_size = input("How many states to generate: ")
    create_initial_state_set(sample_size)