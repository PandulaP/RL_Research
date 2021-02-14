import gym
from gym import wrappers
import custom_cartpole  # custom cart-pole environment
import pbpi.algo_core.algo_file_paths as f_paths

import numpy as np
import pandas as pd

from pathlib import Path

#from scipy.stats import rankdata as rd
from scipy import stats

import io
import base64
import itertools
import tqdm
import os

import sys

###################################################

def create_state_pool():
    """
    Create a pool of initial states to select states by interacting with the environment
    for 10,000 steps.

    Returns:
        tuple (pandas.DataFrame, pandas.DataFrame) : Pandas dataframes having state value 
                                                      information of the stored initial states
    """

    # Initialize the gym environment and rest it
    env = gym.make('CustomCartPole-v0') 
    env.reset()

    # Empty vectors to store values
    obs_vec = []
    term_vec = []

    # Number of steps to execute in the environment
    step_count = 10000

    # Generate states by executing random actions (following a random policy)
    for _ in range(step_count):
        obs, reward, terminated, _ = env.step(env.action_space.sample())

        obs_vec.append(obs)
        term_vec.append(terminated)

        if terminated:
            env.reset()


    # Process the observed state values
    obs_vec = np.array(obs_vec).reshape(step_count,-1)
    term_vec  = np.array(term_vec ).reshape(step_count,-1)

    # Only pick the 'pendulum angle' and 'angular velocity' values
    obs_pend_angle_velo = obs_vec[:,[2,3]]
    
    # Save the full state space to create final state (w/ cart-position and cart-velocity)
    obs_full = obs_vec[:,:]    

    # Join the state observations with termination flag data
    obs_pend_angle_velo_w_flag = np.concatenate([obs_pend_angle_velo,term_vec],axis = 1)

    # Create a Pandas dataframe with the information
    obs_df = pd.DataFrame(obs_pend_angle_velo_w_flag
                        , columns=['pend_angle','angular_velo','flag'])
    obs_df.reset_index(inplace = True)
    obs_df.columns = ['step','pend_angle','angular_velo','flag']
    obs_df.flag.replace([0,1],['Not-Terminated','Terminated'],inplace=True) # Replace the termination flag values
    
    
    # Create a Pandas dataframe with complete state information
    obs_full_df = pd.DataFrame(obs_full
                        , columns=['cart_position', 'cart_velocity','pend_angle','angular_velo'])
    obs_full_df.reset_index(inplace = True)
    obs_full_df.columns = ['step','cart_position', 'cart_velocity','pend_angle','angular_velo']
    
    return obs_df, obs_full_df

###################################################

def compute_full_state(df_row, full_state_df):
    """
    Generate the 'cart velocity' state value for a given pair of 'pendulum angle' and 'angular velocity' state values
    using the complete pool of state dataset.

    Args:
        pandas.Series : A row of a pandas dataframe which include the 
                            - pendulum angle and angular velocity of a state
                            - upper and lower bounds computed on the above two state values

    Returns:
        tuple : a complete state of the environment
                    - (cart position = 0, cart velocity, pendulum angle, angular velocity)
    """

    pend_angle = df_row[0] # First column contains the generated pendulum angle
    angular_vel = df_row[1] # Second column contains the generated angular velocity

    upper_pend_angle_b = df_row[2]    # Third column contains the computed uppre bound on the pendulum angle
    lower_pend_angle_b = df_row[3]   # Fourth column contains the computed lower bound on the pendulum angle

    upper_angu_vel_b = df_row[4]    # Fifth column contains the computed uppre bound on the angular velocity
    lower_angu_vel_b = df_row[5]    # Fourth column contains the computed lower bound the angular velocity

    cart_vel = np.array(full_state_df.loc[(full_state_df.pend_angle <= upper_pend_angle_b) & \
                                            (full_state_df.pend_angle >= lower_pend_angle_b) & \
                                            (full_state_df.angular_velo <= upper_angu_vel_b) & \
                                            (full_state_df.angular_velo >= lower_angu_vel_b) ,'cart_velocity'].values).mean()

    cart_posi = 0 # set cart-position to be zero

    return (cart_posi, cart_vel, pend_angle, angular_vel)

###################################################

def create_full_state_sample(init_df, full_state_df):
    """
    Generate a dataset with all state value variables.

    Args:
        Pandas.DataFrame : Dataset having 'pendulum angle' and 'angular velocity' state values

    Returns:
        Pandas.DataFrame : Dataset with all four state values (cart position, cart velocity, pendulum angle, angular velocity)
    """

    init_df.loc[:,'pend_angle_b_upper'] = init_df.pend_angle.apply(lambda val: val+0.05)
    init_df.loc[:,'pend_angle_b_lower'] = init_df.pend_angle.apply(lambda val: val-0.05)

    init_df.loc[:,'angular_velo_b_upper'] = init_df.angular_velo.apply(lambda val: val+0.05)
    init_df.loc[:,'angular_velo_b_lower'] = init_df.angular_velo.apply(lambda val: val-0.05)

    # Create the final right-skewewd inititial state set
    full_state_value_pairs = []

    for idx, row in init_df.iterrows():
        full_state_value_pairs.append(compute_full_state(row, full_state_df))

    final_df = pd.DataFrame(np.array(full_state_value_pairs), columns=['cart_position'	,'cart_velocity','pend_angle','angular_velo'])
    
    return final_df

###################################################

def create_initial_state_set(sample_size:int):

    # Generate the pool of states
    obs_df, full_state_df =  create_state_pool()

    # Append the 'episode number' and 'episodic step' values
    # to the states stored in the initial pool of states
    episode_count = []
    episodic_step = []

    epi_step_count = 0
    epi_count = 1

    for idx, row in obs_df.iterrows():
        
        epi_step_count += 1   
            
        if row.flag == 'Terminated':
            episodic_step.append(epi_step_count)
            epi_step_count = 0
            
            episode_count.append(epi_count)
            epi_count +=1
            continue
            
        episode_count.append(epi_count)
        episodic_step.append(epi_step_count)
            
    obs_df = pd.concat([obs_df
                        , pd.Series(episodic_step,name='episodic_step')
                        , pd.Series(episode_count ,name='episode_num')],axis=1)


    # Identify number of steps to select from each episode (after removing 4 states leading to terminal state)
    epi_n_epi_steps_fil_df = obs_df.groupby(['episode_num']).episodic_step.max().reset_index()
    epi_n_epi_steps_fil_df.loc[:,'epi_steps_to_use'] = epi_n_epi_steps_fil_df.episodic_step.apply(lambda val: max(0,val-5))

    obs_df = obs_df.merge(right = epi_n_epi_steps_fil_df.loc[:,['episode_num', 'epi_steps_to_use']]
                        , right_on='episode_num'
                        , left_on = 'episode_num'
                        , how = 'left')


    # Empty list to store the selected rows for the reduced dataset
    filter_df_rows = []

    # Iterate over rows; only select the states/rows accoding to 'epi-steps-to-use' values
    for idx, row in obs_df.iterrows():
        
        if row.episodic_step <= row.epi_steps_to_use:
            filter_df_rows.append(row)
        
    obs_df_reduced = pd.concat(filter_df_rows,axis=1).T
    
    # Check if the initial state set is already created
    init_sample_file_path = Path(f_paths.paths['init_sample_file']+f'init_state_sample_s_{sample_size}.csv')
    if init_sample_file_path.is_file():
        return pd.read_csv(init_sample_file_path)

    else:
        # Draw a random sample of size 'sample_size'  from the 'reduced' initial state pool
        np.random.seed(15)

        idx = np.random.choice(obs_df_reduced.index.values, size=sample_size)
        init_state_sample = obs_df_reduced.loc[obs_df_reduced.index.isin(idx),['pend_angle','angular_velo']]
        init_state_sample.reset_index(drop=True, inplace=True)

        init_state_sample_final = create_full_state_sample(init_state_sample, full_state_df)
        init_state_sample_final.to_csv(init_sample_file_path, index=False)

        return init_state_sample_final


if __name__ == '__main__':
    sample_size = input("How many states to generate: ")
    create_initial_state_set(sample_size)