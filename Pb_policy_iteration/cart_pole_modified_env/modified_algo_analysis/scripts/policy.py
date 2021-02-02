import numpy as np
from scipy.stats import rankdata as rd
import torch


########################################
### Derived policy using LabelRanker ###

class Policy():
    
    """
    Description:
    
        - This Policy object takes a given neural network (LabelRanker) model and uses it to define a policy for the agent to follow
    """
    
    def __init__(self, action_space, model, probs):
        self.action_space = action_space # action space of the current environment
        self.model = model               # trained NN (LabelRanker) model
        self.probs = probs               # list of probabilities for actions
        
    def label_ranking_policy(self,obs):
        """ Produces an action for a given state based on the LabelRanker model prediction
            Note: only the pendulum-angle and pendulum-velocity of the input state are considered when producing an action
        
            At each input state:
                - Highest ranked action is selected with a prob. of 0.95
                - Second highest ranked action is selected with a prob. of 0.04
                - Any remaining actions are selected with an equal proabability of .01 """


        # only select the pendulum-velocity and angle from the input state vector
        #state_obs = np.array([obs[2].reshape(-1)[0],obs[3].reshape(-1)[0]]) 
        #state_obs = np.array([round(obs[2].reshape(-1)[0],6),round(obs[3].reshape(-1)[0],6)]) # rounded input
        state_obs = np.array([obs[2].reshape(-1)[0],obs[3].reshape(-1)[0]])
        
        #state_obs = state_obs.reshape(-1,state_obs.shape[0]) # reshape to be a 2D array
        state_obs = torch.from_numpy(state_obs) # convert to a tensor

        # make ranking predictions for all actions
        with torch.no_grad():
            preds = self.model(state_obs.float()) 

        # rank the indexes of actions (from highest ranked/preferred action to lowest)
        #ranked_action_idx = (-rd(preds.detach().numpy())).argsort()[:preds.shape[1]]
        ranked_action_idx = (-rd(preds.detach().numpy())).argsort()

        
        ### return the selected action ###
        
        # if there are more than 2 actions
        if len(self.action_space)>2:
            
            # compute the probabilities for the 3rd action onward
            remain_probs = .00/len(ranked_action_idx[2:])
            n_remain_actions = ranked_action_idx.shape[0]-2

            # since we add random noise to action, policy becomes stochastic (even if we select the 1st ranked action always)
            # select one of the remaining actions 1% time
            action = np.random.choice(ranked_action_idx,1 , p=[self.probs[0], self.probs[1]] + list(np.repeat(remain_probs,n_remain_actions)))[0]
        
        else:
            
            # if there are only 2 actions: select highest preferred actions 95% and 5% of the time
            action = np.random.choice(ranked_action_idx,1 , p=[self.probs[0], self.probs[1]])[0]
        
        # when action space is partitioned, return the corresponding action
        # - a uniform noise term is added to action signals to make all state transitions non-deterministic 
        # clip action value to (-1,1) range
        return np.array([[np.clip(self.action_space[int(action)] + np.array(np.random.uniform(low = -.2,high=.2),dtype=float),-1,1)]])
    

########################################