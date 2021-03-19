import gym
import pbpi.algo_core.algo_file_paths as f_paths # path configurations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


######################################
### Evaluating the learned policy ####

def run_evaluations(policy               # input policy
                    , state_list         # list of initial states
                    , step_thresh = 1000    # step-count (threshold)
                    , env_name = 'CustomCartPole-v0' # name of the environment
                    , simulations_per_state = 100 # number of simulations to generate per state
                    , iterr_num = None # iterration number that the evaluation runs for
                    , print_eval_summary = None # Whether to print the evaluation summary or not
                    , print_policy_behaviour = False # Whether to plot action selection vs. pendulum angle
                    , model_name_input = None # Name of the used LabelRanker model
                    , experiment_run_input = None # At which experiment run the evaluation was called at
                   ):  
                   
    """
    Description:
    
        - For every state in a given list of initial states, 100 simulations are generate and the percentage of
           these simulations that exceeds a predefined step-count threadhold (trajectory length) is computed to measure 
           the performance of the given input policy."""
    

    simu_per_state = simulations_per_state
        
    # create an environment instance
    env_test = gym.make(env_name)
    
    # variable to record the sufficient policy count (across all simulations)
    suf_policy_count = 0
    
    # variable to record episodic returns
    ep_returns = []
    max_return = 0
    min_return = 2000
    
    # iterate over all states in the state list
    for state in state_list:        
        
        # generate 100 simulations from each state
        for _ in range(simu_per_state):
            
            # set the starting state and the current observation to the given state 
            env_test.reset(init_state=state)
            obs = state
        
            # variable to store the return of an episode
            return_ep = 0 

            # execute 1001 steps in the environment
            for _ in range(1001):
                action = policy.label_ranking_policy(obs) # generate action from the policy
                obs, reward, done, _ = env_test.step(action) # execute action
                #obs = observation     # set history
                return_ep += reward   # compute return
                if done: break

            env_test.close()

            # append the return of the episode
            ep_returns.append(return_ep)
            
            # update the max and min return variables
            max_return = max(max_return,return_ep)
            min_return = min(min_return,return_ep)
            
            # increment the sufficient policy count if return exceeds given threshold
            # (note: at every step, 1 reward is produced in the environment)
            if return_ep >= step_thresh:
                suf_policy_count += 1
    

    # Evaluate the policy performance on the neutral starting state, i.e., [0,0,0,0]
    if print_policy_behaviour:

        # Create placeholders for the pendulum angle and action values
        act_vals = []
        pend_angle_vals = []

        # Initialize the environment to starting state
        # Randomly set the initial pendulum value as U[-.1,.1)
        starting_state = [0, 0, np.random.uniform(-.1,.1), 0]
        obs = env_test.reset(init_state = np.array(starting_state))

        # Store the length of the episode
        ret_ep = 0

        # Let the policy interact with the environment
        for _ in range(1001):

            pend_angle_vals.append(obs.reshape(-1)[2]) # append the new pendulum angle value

            a = policy.label_ranking_policy(obs) # generate action from the policy
            obs, r, terminate, _ = env_test.step(a) # execute action
            
            act_vals.append(a.reshape(-1)[0])    # append the performed action value

            ret_ep += r   # compute return (number of executed steps)
            
            if terminate: break

        # Create a dataframe with pendulum angle values and executed actions
        eval_df = pd.DataFrame({'pendulum_angle': pend_angle_vals
                                , 'act_vals': act_vals})
        
        # Add evaluation reward to dataframe
        #eval_df.loc[:,'eval_return'] = ret_ep

        def recode_act_val(val):
            """Recode the action values"""

            if val < 0:
                return f'{abs(val)*50}N force to RIGHT'
            elif val > 0:
                return f'{abs(val)*50}N force to LEFT'
            elif val == 0:
                return f'No force'
            else:
                return 'somethings wrong!'

        eval_df.loc[:, 'Action'] = eval_df.act_vals.apply(lambda val: recode_act_val(val))

        g = sns.displot(x = 'pendulum_angle'
                        , row='Action'
                        , data = eval_df
                        , bins = 100
                        , aspect = 2
                        , height = 3
                        , kde =True).set(xlabel = 'Pendulum Angle')

        g.map(plt.axvline, x=0, c='red')
        g.fig.subplots_adjust(top=.93) 
        g.fig.suptitle('Actions vs. Pendulum angle', fontsize= 10)
        plt.savefig(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_behaviour.png') # save the evaluation image
        plt.show()        
        
        #eval_df.to_csv(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_behaviour.csv', index=False)
        print(f"\nPolicy Iteration: {iterr_num} - Length of the evaluation episode: {ret_ep} (init. state: {[round(val,2) for val in starting_state]})")

    # Evaluation metric returns
    # 1. % sufficient policy counts (total sufficient policies/ total # evaluation runs)
    # 2. 'avg. episodic return'
    # 3. maximum episodic return (across all evaluations)
    # 4. minimum episodic return (across all evaluations)

    avg_return = (sum(ep_returns)/(len(state_list)*simu_per_state))
    pct_sr = (suf_policy_count/(len(state_list)*simu_per_state))*100

    if print_eval_summary:
        
        print(f"\nPolicy Iteration: {iterr_num} - Evaluation results:\n \
                        - Avg. return : {avg_return}\n \
                        - Max. return : {max_return}\n \
                        - Min. return : {min_return}\n \
                        - Successful episodes : {pct_sr}% \n")     

    return (suf_policy_count/(len(state_list)*simu_per_state))*100, avg_return, max_return, min_return 

#######################################