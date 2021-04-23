import gym
import pbpi.algo_core.algo_file_paths as f_paths # path configurations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

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
                action, _ = policy.label_ranking_policy(obs) # generate action from the policy
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

        # Generate action probabilities

        # Create the state value space to get action probability distributions
        state_vals_pend_angl  = np.linspace(-.211, .211, 401).round(3)
        state_vals_angl_vel   = np.linspace(-.461, .461, 401).round(3)
        state_val_combi = list(product(state_vals_pend_angl, state_vals_angl_vel))
        state_val_combi = [list(tup) for tup in state_val_combi]
        state_values    = [[0.,0.]+l for l in state_val_combi]

        action_prob_data = []

        for state in state_values:

            act_prob_dict = {}
            selected_action, action_probs = policy.label_ranking_policy(state) # generate action from the policy

            act_prob_dict['pendulum_angle'] = state[2]
            act_prob_dict['angular_velocity'] = state[3]
            act_prob_dict['action_1_prob'] = action_probs[0]
            act_prob_dict['action_2_prob'] = action_probs[1]
            act_prob_dict['action_3_prob'] = action_probs[2]
            act_prob_dict['selected_action'] = selected_action

            action_prob_data.append(act_prob_dict)

        action_prob_data_df = pd.DataFrame(action_prob_data)
        #action_prob_data_df.to_csv('delete_this.csv', index=False)

        action_prob_data_df.pendulum_angle = action_prob_data_df.pendulum_angle
        action_prob_data_df.angular_velocity = action_prob_data_df.angular_velocity

        cols = ['action_1_prob',	'action_2_prob',	'action_3_prob']

        action_prob_data_df_2 = action_prob_data_df.loc[:,cols]\
                                .subtract(action_prob_data_df.loc[:,cols].min(axis=1), axis=0)\
                                .divide(action_prob_data_df.loc[:,cols].max(axis=1) - action_prob_data_df.loc[:,cols].min(axis=1), axis=0)

        action_prob_data_df = action_prob_data_df.loc[:,['pendulum_angle',	'angular_velocity']]\
                                .merge(action_prob_data_df_2, right_index=True, left_index=True, how='inner')
        
        plt.close("all") # close any plots open already

        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize= (26,6))

        heat1 = action_prob_data_df.pivot_table(index = 'angular_velocity', columns = 'pendulum_angle', values = 'action_1_prob')
        heat2 = action_prob_data_df.pivot_table(index = 'angular_velocity', columns = 'pendulum_angle', values = 'action_2_prob')
        heat3 = action_prob_data_df.pivot_table(index = 'angular_velocity', columns = 'pendulum_angle', values = 'action_3_prob')

        sns.heatmap(heat1, ax=ax[0])
        sns.heatmap(heat2, ax=ax[1])
        sns.heatmap(heat3, ax=ax[2])

        ax[0].set_title("Probability of 'Force to right' action having the highest rank\n", fontsize = 12)
        ax[1].set_title("Probability of 'No Force' action having the highest rank\n", fontsize = 12)
        ax[2].set_title("Probability of 'Force to left' action having the highest rank\n", fontsize = 12)

        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        ax[2].invert_yaxis()

        ax[0].set_xlabel('Pendulum angle')
        ax[1].set_xlabel('Pendulum angle')
        ax[2].set_xlabel('Pendulum angle')

        ax[0].set_ylabel('Angular velocity')
        ax[1].set_ylabel('Angular velocity')
        ax[2].set_ylabel('Angular velocity')

        plt.show()
        fig.savefig(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_act_cond_dist.png') # save the evaluation image

        # Create placeholders for the pendulum angle, angular velocity and action values
        act_vals = []
        pend_angle_vals = []
        angular_vel_vals = []

        # Initialize the environment to starting state
        # Randomly set the initial pendulum value as U[-.1,.1)
        starting_state = [0, 0, np.random.uniform(-.1,.1), 0]
        obs = env_test.reset(init_state = np.array(starting_state))

        # Store the length of the episode
        ret_ep = 0

        # Let the policy interact with the environment
        for _ in range(1001):

            pend_angle_vals.append(obs.reshape(-1)[2]) # append the new pendulum angle value
            angular_vel_vals.append(obs.reshape(-1)[3]) # append the new angular veloocity value

            a, _ = policy.label_ranking_policy(obs) # generate action from the policy
            obs, r, terminate, _ = env_test.step(a) # execute action
            
            act_vals.append(a.reshape(-1)[0])    # append the performed action value

            ret_ep += r   # compute return (number of executed steps)
            
            if terminate: break

        # Create a dataframe with pendulum angle values and executed actions
        eval_df = pd.DataFrame({'pendulum_angle': pend_angle_vals
                                , 'angular_velocity': angular_vel_vals
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

        melted_eval_df = eval_df.drop(columns='act_vals')

        melted_eval_df = melted_eval_df.melt(id_vars='Action'
                                            , var_name = 'state_variable'
                                            , value_name = 'vals')

        # dis_p = sns.displot(x = 'vals'
        #                 , col = 'state_variable'
        #                 , row = 'Action'
        #                 , data = melted_eval_df
        #                 , bins = 100
        #                 , aspect = 2
        #                 , height = 3
        #                 , kde =True)#.set(xlabel = 'Pendulum Angle')


        # dis_p.map(plt.axvline, x=0, c='red')
        # dis_p.fig.subplots_adjust(top=.93) 
        # dis_p.fig.suptitle('Actions vs. Pendulum angle & Angular Velocity', fontsize= 10)
        # dis_p.savefig(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_bhvior_1.png') # save the evaluation image

        try:
            j_plot = sns.jointplot(data = eval_df
                                , x = "pendulum_angle"
                                , y = "angular_velocity"
                                , hue = "Action"
                                , hue_order = sorted(eval_df.Action.unique(), reverse=True)
                                , palette = ['orange', 'blue', 'brown'] if len(eval_df.Action.unique()) == 3 else ['orange', 'blue', 'brown', 'green', 'pink'][:len(eval_df.Action.unique())] 
                                , kind = "kde"
                                , height = 7
                                )

            j_plot.fig.suptitle('Pendulum angle & Angular Velocity', fontsize= 10)
            j_plot.fig.subplots_adjust(top=.93) 
            j_plot.savefig(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_bhvior_2.png') # save the evaluation image
            
            #plt.savefig(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_behaviour.png') # save the evaluation image
            plt.show()

        except:
            print(f"\nCan't create joint-plot: Matrix is not positive definite!\n")
            pass
        
        fig2, ax2 = plt.subplots(nrows = 2
                            , ncols = 1
                            , figsize = (15, 12))

        sns.scatterplot(data = eval_df
                        , x = eval_df.index
                        , y = 'pendulum_angle'
                        , hue = 'Action'
                        , hue_order = sorted(eval_df.Action.unique(), reverse=True)
                        , palette = ['orange', 'blue', 'brown'] if len(eval_df.Action.unique()) == 3 else ['orange', 'blue', 'brown', 'green', 'pink'][:len(eval_df.Action.unique())] 
                        , ax =  ax2[0])
        ax2[0].set_xlabel('Step of the episode')
        ax2[0].set_title('Performed action at different pendulum angles at each step of the episode')

        sns.scatterplot(data = eval_df
                        , x = eval_df.index
                        , y = 'angular_velocity'
                        , hue = 'Action'
                        , hue_order = sorted(eval_df.Action.unique(), reverse=True)
                        , palette = ['orange', 'blue', 'brown'] if len(eval_df.Action.unique()) == 3 else ['orange', 'blue', 'brown', 'green', 'pink'][:len(eval_df.Action.unique())]
                        , ax =  ax2[1])
        ax2[1].set_title('Performed action at different angular velocity at each step of the episode')
        ax[1].set_xlabel('Step of the episode')

        plt.show()
        fig2.savefig(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_bhvior_3.png') # save the evaluation image
        
        #sns.scatterplot(data = eval_df, x = eval_df.index, y = 'Action', ax =  ax[2])

        #eval_df.to_csv(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_behaviour.csv', index=False)
        print(f"\nPolicy Iteration: {iterr_num} - Length of the evaluation episode: {ret_ep} (init. state: {[round(val,2) for val in starting_state]})\n")

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