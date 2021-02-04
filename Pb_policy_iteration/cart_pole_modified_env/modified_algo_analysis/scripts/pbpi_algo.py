########################################
### importing the necessary packages ###

import gym
from gym import wrappers
import custom_cartpole  # custom cart-pole environment

import numpy as np
import pandas as pd

from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

########################################


#####################################
### preference generation process ###

def evaluate_preference(starting_state # starting state of roll-outs
                        , action_1     # first action to execute at the starting-state
                        , action_2     # second action to execute at the starting state
                        , policy_in    # policy to folow
                        , environment_name = 'CustomCartPole-v0'   # name of the environment
                        , discount_fac = 1        # discounting factor
                        , n_rollouts = 20         # number of roll-outs to generate per action
                        , max_rollout_len = 1500  # maximum length of a roll-out
                        , label_ranker = False    # whether to use the label-ranking model or not
                        , p_sig = 0.05            # p-value to use for t-test (to compare returns of roll-outs)
                        , tracking = False
                        ):
    
    """
    Description:
    
        - Roll-outs are generated at each state in the initial state set by starting from the given input action 
          and following the given policy afterwards. 
        - Returns of the roll-outs are used to generate preferences for the input action pair.
        - Generated preferences are returned to be create a training dataset to learn the LabelRanker model.    
    """
    
    # initializing variables
    policy = policy_in          
    n_rollouts = n_rollouts     
    gamma = discount_fac    
    s_init = starting_state
    max_traj_len = max_rollout_len 
        
    # we store the num. actions executed within the evaluation process (to measure complexity)
    action_count = 0 
        
    # dictionary to store input action values
    actions = { 'one' : action_1    
              , 'two' : action_2}    

    # dictionary to store rewards of roll-outs
    r = { 'one' : [None]*n_rollouts 
        , 'two' : [None]*n_rollouts}  

    # dictionary to store average discounted return for each action
    avg_r = {}  
    
    # select each action of the input actions to generate roll-outs:
    for action_key, action_value in actions.items():

        # generate the defined number of roll-outs for selected action
        for rollout in range(n_rollouts):

            # create an environment object and set the starting state to the input (initial) state
            env = gym.make(environment_name)
            env.reset(init_state=s_init) # modified env.reset() in custom env: it accepts a starting state

            # genereate random noice for action
            rand_act_noice =  np.array([[np.random.uniform(low = -.2,high=.2)]])
                                            
            # apply the action (custom environment accepts float actions)
            observation, reward, done, _ = env.step(np.clip(action_value + rand_act_noice,-1,1)) # clip action value to (-1,1) range
            
            # define the history variable to store the last observed state
            hist = observation 
            
            # add the immediate reward received after executing the action
            r[action_key][rollout] = reward  

            # follow the given policy to generate a roll-out trajectory 
            traj_len = 1
            while traj_len < max_traj_len and not done: 
                
                # sample next state using the label-ranking model (if TRUE)
                if label_ranker: 
                    observation, reward, done, _ = env.step(policy.label_ranking_policy(hist))
                    
                    # replace current history with the observed state
                    hist = observation
                    action_count+=1
                
                # sample next state using a random policy
                else: 
                    observation, reward, done, _ = env.step(policy(env))
                    action_count+=1

                # compute discounted-reward at each step of the roll-out and store the roll-out return
                r[action_key][rollout] += (gamma**traj_len) * reward

                traj_len += 1

            # close the environment after creating roll-outs
            env.close()
            del env
        
        # calculate the average discounted returns of the two actions
        avg_r[action_key]  = sum(r[action_key]) / len(r[action_key])

    # run a t-test to check whether the observed difference between average returns is significant
    # (unpaird t-tests: equal variance)
    t_val, p_val = stats.ttest_ind(r['one'], r['two']) 
    
    # track output
    if tracking:
        print(f"state: {[state_dim.reshape(-1)[0] for state_dim in [s_init[2],s_init[3][0][0]]]} | a_j(R): {avg_r['one']} | a_k(R): {avg_r['two']} | sig: {'Yes' if (p_val <= p_sig) else '--'}")
    
    # return preference information
    if (avg_r['one'] > avg_r['two']) and (p_val <= p_sig):
        return {'state': s_init
               , 'a_j' : actions['one']
               , 'a_k' : actions['two']
               , 'preference_label' : 1}, action_count
    
    elif(avg_r['one'] < avg_r['two']) and (p_val <= p_sig):
        return {'state': s_init
               , 'a_j' : actions['one']
               , 'a_k' : actions['two']
               , 'preference_label' : 0}, action_count
    
    # return NaN if avg. returns are not significantly different from each other OR are equal
    else: 
        return {'state': np.nan
               , 'a_j' : np.nan
               , 'a_k' : np.nan
               , 'preference_label' : np.nan}, action_count
    
#####################################

##########################################
### LabelRanker Model training process ###

def train_model(train_data                  # collection of all preference data
                , action_space              # action space of the task
                , model_name:str            # name for the model (to store)
                , batch_s = 4               # batch size to train the NN model
                , mod_layers = [10]         # model configuration
                , n_epochs = 1000           # num. of epochs to train the model
                , l_rate = 0.01             # learning rate for the optimization process  
                , show_train_plot = False   # flag to display the 'training-loss vs. epoch' plot
                , show_dataset = False):    # flag to display the training dataset
    
    
    """
    Description:
    
        - This function process all preference data to construct a training dataset for the LabelRanker model.
        - One training sample takes the form:
            X: [state-value (2-D)]
            Y: [(normalized) ranking of actions (n-D)], where 'n' is the number of actions in the action space.
        - For a given (2-D) state input, the (trained) model, i.e., LabelRanker, predicts the rank of 
           all possible actions at the input state 
    """

    
    ### creating the training dataset ###
        
    # convert training data input to a dataframe | 
    # remove the rows that have NaN, i.e.,preference evaluations without any action preference
    train_df = pd.DataFrame(train_data).dropna()

    # create a key for each state in the dataset
    # (only select the 'pendulum-velocity & pendulum-angle)
    #train_df.loc[:, 'state_key'] = train_df.state.apply(lambda x: x[2].astype(str)+"_"+x[3].astype(str))
    #train_df.loc[:, 'state_key'] = train_df.state.apply(lambda x: round(x[2].reshape(-1)[0],6).astype(str)+"_"+round(x[3].reshape(-1)[0],6).astype(str))
    train_df.loc[:, 'state_key'] = train_df.state.apply(lambda x: x[2].reshape(-1)[0].astype(str)+"_"+x[3].reshape(-1)[0].astype(str))

    
    # ******************************** # EXPERIMENTAL STEP START
    # create a full state key (state+action preference)
    #train_df.loc[:, 'state_action_key'] = train_df.state.apply(lambda x: round(x[2],6).astype(str)+"_"+round(x[3],6).astype(str)) +"_"+ train_df.a_j.apply(lambda x: x[0][0].astype(str))+"_"+ train_df.a_k.apply(lambda x: x[0][0].astype(str)) 

    
    # drop duplicates (if one training-set maintained) : only keep the first learned preference
    #train_df.drop_duplicates(subset=['state_key'], keep='first', inplace=True)
    #train_df.drop_duplicates(subset=['state_action_key'], keep='first', inplace=True)
    
    #train_df.drop(columns=['state_action_key'], inplace=True)
    
    # ******************************** # EXPERIMENTAL STEP END
    
    # check if the training dataset is empty 
    # (if empty, subsequent steps have to be skipped)
    if not(train_df.shape[0]>0):
        
        # if training dataset is emtpy - return None (break the training loop)
        return None
    
    else:
        
        ### computing action-preference counts for every action (for every states) ###
        
        # identify the 'prefered-action' at each 'state, action-pair' preference evaluation
        train_df.loc[:,'prefered_action'] = train_df.apply(lambda row: row['a_j'][0][0] if row['preference_label'] == 1 else row['a_k'][0][0]  ,axis=1)

        # compute the number of times each action is prefered at each state
        action_preference_counts = train_df.groupby('state_key').prefered_action.value_counts().unstack()
        action_preference_counts.replace(np.nan,0,inplace=True) # if an action is not preferred at a state, set pref. count to '0'

        # remove the column index names of the `action_preference_counts' summary table
        action_preference_counts.columns.name = None

        # find any action(s) that was not preferred at all sampled states 
        # - this is important because a ranking for every possible action
        #   at each state needs to be included in the training (label) data
        missed_actions = [action for action in action_space if action not in action_preference_counts.columns.tolist()]
        missed_actions = np.array(missed_actions).astype(action_preference_counts.columns.dtype) # convert to the same data-type of remaining columns

        # add any missing actions to the `action_preference_counts' table
        if len(missed_actions)>0:

            # add the missing action (with a preference count of zero)
            for action in missed_actions:
                action_preference_counts.loc[:,action] = 0

            # sort the actions in the summary according to arrangement in action space (ascending order)
            action_preference_counts = action_preference_counts.reindex(sorted(action_preference_counts.columns), axis=1)    

        
        # convert the action-preference-counts (of actions at each state) to a vector and add it as a new column
        #  - data in this column is used to create training labels
        action_preference_counts.loc[:, 'preference_label_vector'] = pd.DataFrame({'label_data': action_preference_counts.iloc[:,0:].values.tolist()}).values

        # append the column having action-preference-counts vectors to the training dataset
        train_df = train_df.merge(right = action_preference_counts.loc[:,['preference_label_vector']]
                                  , right_index= True
                                  , left_on = 'state_key'
                                  , how = 'left')
        

        # create the reduced training dataset 
        # - drop unnecessary columns & duplicate rows (which have duplicate data for same states)
        train_df_reduced = train_df.loc[:,['state', 'state_key', 'preference_label_vector']]
        train_df_reduced.drop_duplicates(subset=['state_key'],inplace=True)
        train_df_reduced.preference_label_vector = train_df_reduced.preference_label_vector.apply(lambda row: np.array(row).astype(np.float)) # convert all label vectors to float
        
        if show_dataset:
            print(f'\nTraining data samples: {train_df_reduced.shape[0]}')
            print(train_df_reduced.loc[:,['state_key', 'preference_label_vector']])
        
        ### preparing the training dataset for the neural network (LabelRanker) model ###

        # normalize the action-preference-counts vectors (label data for the model)
        # - this step produces the rankings:
        # - i.e., the action(s) with the highest preference count(s) will have the highest value(s)
        # - after normalization
        output_labels_temp = np.array(train_df_reduced.preference_label_vector.tolist())
        row_sums = output_labels_temp.sum(axis=1)
        output_labels_normalized = output_labels_temp / row_sums[:, np.newaxis]
        output_labels = torch.from_numpy(output_labels_normalized) # convert to tensor

        # Generate the input state data tensors (feature data for the model)
        #   This only includes 'pendulum-angle' and 'angular velocity'
        #   State values are rounded; this seems to improve the performance
        input_states  = torch.from_numpy(np.array(train_df_reduced.state.apply(lambda x: [round(x[2].reshape(-1)[0],5).astype(float), round(x[3].reshape(-1)[0],5).astype(float)]).tolist())) # only select pole-position and pole-velocity

        
        # create TensorDataset
        train_ds = TensorDataset(input_states , output_labels)
        
        # define the batch size
        batch_size = batch_s 
        
        # define the data loader
        train_dl = DataLoader(train_ds
                              , batch_size
                              , shuffle=True
                              #, drop_last=True
                             )
        
        
    ### defining and training the neural network (LabelRanker) model ###        
    
    class Model(nn.Module):

        def __init__(self, input_state_len, output_label_len, layers, p=0.3):

            super(Model,self).__init__()

            all_layers = []
            input_size = input_state_len

            # create layers
            for layer_dim in layers:
                all_layers.append(nn.Linear(input_size, layer_dim))
                all_layers.append(nn.LeakyReLU(inplace=True))
                #all_layers.append(nn.BatchNorm1d(layer_dim))
                #all_layers.append(nn.Dropout(p))
                input_size = layer_dim

            all_layers.append(nn.Linear(layers[-1], output_label_len))

            self.layers = nn.Sequential(*all_layers)

        def forward(self, state_vec):
            x = self.layers(state_vec)
            return x

        
    # create a NN model instance
    model = Model(input_states.shape[1], output_labels.shape[1], mod_layers)

    # define optimizer and loss
    #opt = torch.optim.SGD(model.parameters(), lr = l_rate)
    opt = torch.optim.Adam(model.parameters(), lr = l_rate, weight_decay = 0.00001)
    loss_fn = F.mse_loss

    # list to store losses
    aggregated_losses = []

    # defining a function to train the model
    def fit(num_epochs, model, loss_fn, opt):
        
        for _ in range(num_epochs):
            for xb,yb in train_dl:

                # Generate predictions
                pred = model(xb.float())
                loss = loss_fn(pred, yb.float())

                # Perform gradient descent
                loss.backward()
                opt.step()
                opt.zero_grad()

            aggregated_losses.append(loss_fn(model(input_states.float()), output_labels.float()).detach().numpy())

        #print('\nTraining loss: ', loss_fn(model(input_states.float()), output_labels.float()).detach().numpy(),'\n')
        
        # return training loss
        return loss_fn(model(input_states.float()), output_labels.float()).detach().numpy()
    

    # train the model
    epochs = n_epochs
    loss_v = fit(epochs, model, loss_fn, opt)

    # save the trained model
    PATH = f"../data/output/models/{model_name}_pbpi_model.pt"
    torch.save(model.state_dict(), PATH)
    
    # plot the model loss
    if show_train_plot:
        plt.plot(range(epochs), aggregated_losses)
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.title(f'Training samples: {train_df_reduced.shape[0]} | Training loss: {np.round(loss_v,5)}\n')
        plt.show()

    # set the model to evaluation mode and return it
    return model.eval()

##########################################