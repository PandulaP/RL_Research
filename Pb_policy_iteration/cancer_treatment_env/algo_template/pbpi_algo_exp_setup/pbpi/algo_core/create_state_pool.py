import gym
from gym import wrappers
import custom_cartpole  # custom cart-pole environment

import pbpi.algo_core.algo_file_paths as f_paths

import numpy as np
import pandas as pd

from pathlib import Path

###################################################

def create_initial_state_set(sample_size:int, init_state_scenario:bool = False, seed = 16):

    np.random.seed(seed)
    
    if init_state_scenario:
        # Create an initial patient set with tumor-size>1
        tumor_size_init, toxicity_init = np.vstack((np.random.uniform(low=1,high=2, size = sample_size)
                                                    , np.random.uniform(low=0,high=2, size = sample_size)))
        init_patients = [(t_size, toxicity) for t_size, toxicity in zip(tumor_size_init,toxicity_init)]

    else:    
        # Initial patient set
        tumor_size_init, toxicity_init = np.random.uniform(low=0, high=2, size = (2, sample_size))
        init_patients = [(t_size, toxicity) for t_size, toxicity in zip(tumor_size_init,toxicity_init)]
    
    return init_patients


if __name__ == '__main__':
    sample_size = input("How many states to generate: ")
    create_initial_state_set(sample_size)