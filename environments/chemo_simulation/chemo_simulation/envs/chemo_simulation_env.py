# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
"""
A simple chemotherapy mathematical model proposed by Zhao et al. (2009).
This environment adopts the OpenAI gym and it can be used to generate virtual patients and clinical trial data.
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class ChemoSimulationEnv(gym.Env):
    
    """
    Description:
        This model captures the below factors in cancer treatment:
           - Tumor growth in the absence of chemotherapy.
           - Patients’ negative wellness outcomes in response to chemotherapy.
           - The drug’s capability for killing tumor cells while also increasing toxicity.
           - The interaction between tumor cells and patient wellness.
           
        Two state variables, the 'tumor size' and the 'toxicity' of a patient are modeled using 
        a system of ordinary difference equations proposed by Zhao et al. (2009). Simulation starts
        with a non-zero tumor size for the patient at initialization and the chemotherapy 
        treatment (dose) is given at discrete steps, where each step denotes the number of months 
        after the start of the treatment (0, 1, 2,..., n). The action applied at each step
        represents the dosage level and it is a number between 0 (min) and 1 (max). The possible 
        death of a patient in the course of a treatment is modeled by means of a hazard rate model.
        The goal of the environment is to is to learn an optimal treatment policy mapping states 
        to actions in the form of a dosage level.
        
    Source:
        This environment is based on the simple chemotherapy mathematical model 
        proposed by Zhao et al. (2009).
   
    Observation:
        Type: Box(2)
        Num    Observation                  Min    Max
        0      Tumor size                   0      10
        1      Patient's negative wellness  0      10
        
    Action:
        Type: Box(1)
        Num    Action                 Min    Max     
        0      Chemotherapy Dosage    0      1
        
    Reward:
        Reward is 1 for every step, including the termination step.
        
    Starting State:
        All observations are assigned a uniform random value in [0.0, 2.0].
        Possible to pass an initial state when required.
    
    Episode Termination:
        As of now, there is no termination condition in place since the original model in Zhao et al. (2009) 
        does not specify one.
        It is possible to define termination conditions based on:
            - the probability of death of a patient between two successive treatments (e.g., P(death) > 0.99),
            - tumor size at a given time point (e.g., tumor size < 0).
        
    """
    
    def __init__(self):
        
        
        # Transition functions parameters
        self.a1 = 0.1
        self.a2 = 0.15
        self.b1 = 1.2
        self.b2 = 1.2

        self.d1 = 0.5
        self.d2 = 0.5 
        
        self.m_0 = None # Tumor size at time '0'
        self.w_0 = None # Initial value of patient's (negative) wellness
        
        # Hazard function paremeters
        self.mu_0 = -4
        self.mu_1, self.mu_2 = 1,1 # To denote both tumor size and toxicity have equal influence on patient’s survival.

        # Termination condition on patient's probabilty of death
        self.death_prop_threshold = 1
        
        # Chemotherapy dose levels: possible values in the range [0,1]
        self.action_space = spaces.Box(low=0, high=1, shape = (1,), dtype=np.float32)
    
        # State space: Tumor size and patient's (negative) wellness
        self.observation_space = spaces.Box(low = np.array([0,0]), high = np.array([10,10]) , dtype=np.float32)
    
        self.seed()
        self.state = None  
    
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def step(self, action):
        
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        m_t, w_t = self.state
        
        # Given chemotherapy dose: acceptable range [0,1]
        d_t = np.clip(action,a_min=0,a_max=1) 
        
        # Indicator func. for the tumor size 
        # If tumour size = 0, we consider patient to be completely cured (tumor won't grow again)
        ind_m_t = np.sign(m_t) 

        # Indicator transition functions for tumor size and patient's wellness
        w_dot_t = self.a1 * max(m_t, self.m_0) + self.b1 * (d_t - self.d1)
        m_dot_t = (self.a2 * max(w_t, self.w_0) - self.b2 * (d_t - self.d2)) * ind_m_t

        # Compute the next state values
        m_t_new = m_t + m_dot_t
        w_t_new = w_t + w_dot_t

        # Tumor size cannot be less than 0
        self.state = (np.clip(m_t_new,0,100), w_t_new)
        
        # Hazard rate model
        lambda_t     = np.exp(self.mu_0 + self.mu_1*w_t + self.mu_2 * m_t)
        lambda_t_new = np.exp(self.mu_0 + self.mu_1*w_t_new + self.mu_2 * m_t_new)
        
        # Patient's probability of death: sum of Hazard rate model values for (t-1,t] time steps
        #  (since in discrete time steps, integration is equal to the summation).
        p_death = 1 - np.exp(-(lambda_t+lambda_t_new))
        
        
        # Check for termination condition
        # As of now, there is no termination condition in place
        # Prob. death == 1
        done = bool(p_death >= np.round(self.death_prop_threshold,2)) 
    
        if not done:
            reward = 1.0
        else:
            reward = 0.0
            logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
        
        return self.state, reward, done, p_death
    
    def reset(self, init_state=None):
        
        if init_state is not None:
            # Accept an initial state at reset
            self.state = init_state 
        else:
            # Initial tumor size and patient's wellness are generated from independent uniform (0,2) deviates. 
            self.state = self.np_random.uniform(low=0, high=2, size = (2,))
        
        # Set the patient's tumor size and (negative) well-being at initialization 
        self.m_0, self.w_0 = self.state
        
        return np.array(self.state)
