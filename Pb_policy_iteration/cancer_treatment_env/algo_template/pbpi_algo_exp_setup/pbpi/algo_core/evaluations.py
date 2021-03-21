import gym
import pbpi.algo_core.algo_file_paths as f_paths # path configurations
from pbpi.algo_core.create_state_pool import create_initial_state_set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


######################################
### Evaluating the learned policy ####

def run_evaluations(policy               # input policy
                    , state_list         # list of initial states
                    , virtual_patients = 200    # step-count (threshold)
                    , env_name    = 'ChemoSimulation-v0' # name of the environment
                    , simulations_per_state = 1 # number of simulations to generate per state
                    , sim_episode_length =  6 # number of steps to continue in one episode
                    , iterr_num          = None # iterration number that the evaluation runs for
                    , print_eval_summary = None # Whether to print the evaluation summary or not
                    , print_policy_behaviour = False # Whether to plot action selection vs. pendulum angle
                    , model_name_input     = None # Name of the used LabelRanker model
                    , experiment_run_input = None # At which experiment run the evaluation was called at
                    , seed = 16
                   ):  
                   
    """
    Description:
    
        - For every state in a given list of initial states, 100 simulations are generate and the percentage of
           these simulations that exceeds a predefined step-count threadhold (trajectory length) is computed to measure 
           the performance of the given input policy."""
    

    simu_per_state = simulations_per_state

    # Create 200 virtual patients
    INIT_STATES = create_initial_state_set(virtual_patients, seed = seed)
        
    # create an environment instance
    env_test = gym.make(env_name)
    
    # Variables to record the avg. tumor size and avg. max. toxicity and prob. of death values 
    # across all virtual parients at the end of rollouts
    avg_t_size       = 0
    avg_max_toxicity = 0
    avg_p_death      = 0
    
    # variable to record episodic performance values
    ep_t_size       = 0
    ep_max_toxicity = -1_000
    ep_p_death      = 0
    
    # iterate over all states in the state list
    for state in state_list:        
        
        # generate simulations from each state
        for _ in range(simu_per_state):
            
            # set the starting state and the current observation to the given state 
            env_test.reset(init_state=state)
            obs = state
        
            # # variable to store the return of an episode
            # return_ep = 0 

            # execute steps in the environment
            for _ in range(sim_episode_length):
                action = policy.label_ranking_policy(obs) # generate action from the policy
                obs, reward, done, p_death = env_test.step(action) # execute action
                #obs = observation     # set history
                #return_ep += reward   # compute return

                # Record the episode observations
                ep_t_size  = obs[0]
                ep_max_toxicity = max(ep_max_toxicity, obs[1])
                ep_p_death = p_death

                if done: break

            env_test.close()

            # # append the return of the episode
            # ep_returns.append(return_ep)
            
            # # update the max and min return variables
            # max_return = max(max_return,return_ep)
            # min_return = min(min_return,return_ep)
            
            # # increment the sufficient policy count if return exceeds given threshold
            # # (note: at every step, 1 reward is produced in the environment)
            # if return_ep >= step_thresh:
            #     suf_policy_count += 1
        
        # Add the episodic eval. metric values to the totals
        avg_t_size = avg_t_size + ep_t_size
        avg_max_toxicity = avg_max_toxicity + ep_max_toxicity
        avg_p_death = avg_p_death + ep_p_death


    # Compute the averages of the metric values
    avg_t_size = avg_t_size/len(state_list)
    avg_max_toxicity = avg_max_toxicity/len(state_list)
    avg_p_death = avg_p_death/len(state_list)


    learned_policy_dic = {'max_toxicity': avg_max_toxicity
                        , 'end_tumor_size': avg_t_size
                        ,  'prob_death': avg_p_death
                        }

    lrned_p_metrics_df = pd.DataFrame(learned_policy_dic, index=['learned'])   

    def generate_evaluation_plot(init_state_set=INIT_STATES, learned_policy_metrics_df = lrned_p_metrics_df):

        # Create instance of the environment
        env = gym.make('ChemoSimulation-v0')

        # Dosage levels to test
        dosages =    {'low': np.array([0.1])
                    , 'mid': np.array([0.4])
                    , 'rand': None
                    , 'high': np.array([0.7])
                    , 'extreme': np.array([1.])}


        # Select the initial set of patients
        selected_patients = init_state_set

        # Dict to store patient information
        patient_dic = {}

        patient_count = 0

        # Run test for patients
        for patient in selected_patients:
            
            # Dict to store observations & prob. of death
            obs_dic = {}
            p_death_dic = {}
            
            # Run for each dosage level separately
            for level, dosage in dosages.items():

                # Create patient: initial Tumor size and wellness
                starting_state = np.array([patient[0]]),np.array([patient[1]])
                
                # St
                obs_dic[level] = []
                obs_dic[level].append(starting_state)

                # Simulate the environment
                env.reset(init_state = starting_state)

                if level == 'rand':
                    # Run treatment for 6 months
                    for _ in range(7):
                        obs, reward, done, p_death = env.step(np.array([np.random.choice(np.array([0.1,0.4,0.7,1.]))]))
                    
                        obs_dic[level].append(obs)
                        p_death_dic[level] = p_death
                        
                        if done:
                            break

                else:
                    # Run treatment for 6 months
                    for _ in range(7):
                        obs, reward, done, p_death = env.step(dosage)
                        
                        obs_dic[level].append(obs)
                        p_death_dic[level] = p_death
                        
                        if done:
                            break

                env.close()

        patient_dic[patient_count] = obs_dic, p_death_dic
        patient_count += 1
            
        ##############################################

        # Select data for the plot from each patient
        def get_stats(dic_vals):
            h_tox = np.array(dic_vals)[:,1].max()
            t_size = np.array(dic_vals)[-1,0][0]
            
            return h_tox, t_size

        # Dicts to store patient data during/at the end of treatment
        patient_data = {}
        patient_pdeath_data = {}

        for patient, patient_obs in patient_dic.items():
            
            plot_data = {}
            _pdeath_data = {}

            for level, obs in patient_obs[0].items():
                plot_data[level] = get_stats(obs)
                
            for level, pdeath in patient_obs[1].items():
                _pdeath_data[level] = pdeath

            patient_data[patient] = plot_data
            patient_pdeath_data[patient] = _pdeath_data


        # Aggregate data from every patient
        df_li = []
        for patient, plot_data in patient_data.items():
            tmp_df = pd.DataFrame(plot_data).T
            tmp_df.columns = ['max_toxicity', 'end_tumor_size']
            tmp_df.loc[:,'Patient'] = patient
            df_li.append(tmp_df)

        # Compute the averages
        plot_df = pd.concat(df_li,axis=0)
        plot_df.reset_index(inplace=True)

        plot_df_summary = plot_df.groupby(['index'])[['max_toxicity','end_tumor_size']].mean().reset_index()
        plot_df_summary.sort_values('end_tumor_size',inplace=True)
        plot_df_summary.set_index('index',inplace=True)

        # Compute prob. of death for each treatment-type
        pdeath_df = pd.DataFrame(pd.DataFrame(patient_pdeath_data).T.mean(axis=0))
        pdeath_df.columns = ['prob_death']

        # Prepare final eval. metric dataframe
        plot_df_summary = plot_df_summary.merge(right = pdeath_df
                                                , right_index=True
                                                , left_index=True)
        plot_df_summary

        # Join learned policy performance
        plot_df_summary = pd.concat([plot_df_summary, learned_policy_metrics_df])  

        ##########################################################
        ##################### Plotting data #####################

        # Data for the connecting line (excluding random dosage)
        line_y = plot_df_summary.loc[(plot_df_summary.index == 'extreme') \
                                    | (plot_df_summary.index == 'high') \
                                    | (plot_df_summary.index == 'mid') \
                                    | (plot_df_summary.index == 'low'),'max_toxicity'].values

        line_x = plot_df_summary.loc[(plot_df_summary.index == 'extreme') \
                                    | (plot_df_summary.index == 'high') \
                                    | (plot_df_summary.index == 'mid') \
                                    | (plot_df_summary.index == 'low') , 'end_tumor_size'].values     



        # Create the plot
        fig, ax = plt.subplots(nrows =  1, ncols=2, figsize = (14,5))

        sns.scatterplot(x='end_tumor_size'
                    , y= 'max_toxicity'
                    , data = plot_df_summary
                    , s=120
                    , ax = ax[0]
                    )

        for i in range(plot_df_summary.shape[0]):
            ax[0].text(x=plot_df_summary.end_tumor_size[i]+0.2
                    ,y=plot_df_summary.max_toxicity[i]+0.2
                    ,s=plot_df_summary.index[i]
                    , fontdict=dict(color='black', size=11)
                    )

        sns.lineplot(x = line_x
                    , y = line_y
                    , ax = ax[0])

        ax[0].lines[0].set_linestyle("--")
        ax[0].set(ylim=(0, 7))
        ax[0].grid()
        ax[0].set_title ('Max Toxicity vs. Tumor size')
        ax[0].set_ylabel('Max toxicity')
        ax[0].set_xlabel('Tumor size')

        sns.barplot(y = 'prob_death'
                    , x = plot_df_summary.index
                    , data = plot_df_summary
                    , ax =  ax[1]
                    , palette = sns.color_palette("mako"))



        ax[1].set_title ('Probability of Death at Treatment End')
        ax[1].set_ylabel('Probability of death')
        ax[1].set_xlabel('Treatment type')
        plt.savefig(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_behaviour.png') # save the evaluation image
        plt.show()  

    # # Evaluate the policy performance on the random starting state
    # if print_policy_behaviour:

    #     # Create placeholders for the pendulum angle and action values
    #     act_vals = []
    #     pend_angle_vals = []

    #     # Initialize the environment to starting state
    #     # Randomly set the initial pendulum value as U[-.1,.1)
    #     starting_state = [0, 0, np.random.uniform(-.1,.1), 0]
    #     obs = env_test.reset(init_state = np.array(starting_state))

    #     # Store the length of the episode
    #     ret_ep = 0

    #     # Let the policy interact with the environment
    #     for _ in range(1001):

    #         pend_angle_vals.append(obs.reshape(-1)[2]) # append the new pendulum angle value

    #         a = policy.label_ranking_policy(obs) # generate action from the policy
    #         obs, r, terminate, _ = env_test.step(a) # execute action
            
    #         act_vals.append(a.reshape(-1)[0])    # append the performed action value

    #         ret_ep += r   # compute return (number of executed steps)
            
    #         if terminate: break

    #     # Create a dataframe with pendulum angle values and executed actions
    #     eval_df = pd.DataFrame({'pendulum_angle': pend_angle_vals
    #                             , 'act_vals': act_vals})
        
    #     # Add evaluation reward to dataframe
    #     #eval_df.loc[:,'eval_return'] = ret_ep

    #     def recode_act_val(val):
    #         """Recode the action values"""

    #         if val < 0:
    #             return f'{abs(val)*50}N force to RIGHT'
    #         elif val > 0:
    #             return f'{abs(val)*50}N force to LEFT'
    #         elif val == 0:
    #             return f'No force'
    #         else:
    #             return 'somethings wrong!'

    #     eval_df.loc[:, 'Action'] = eval_df.act_vals.apply(lambda val: recode_act_val(val))

    #     g = sns.displot(x = 'pendulum_angle'
    #                     , row='Action'
    #                     , data = eval_df
    #                     , bins = 100
    #                     , aspect = 2
    #                     , height = 3
    #                     , kde =True).set(xlabel = 'Pendulum Angle')

    #     g.map(plt.axvline, x=0, c='red')
    #     g.fig.subplots_adjust(top=.93) 
    #     g.fig.suptitle('Actions vs. Pendulum angle', fontsize= 10)
    #     plt.savefig(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_behaviour.png') # save the evaluation image
    #     plt.show()        
        
    #     #eval_df.to_csv(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_behaviour.csv', index=False)
    #     print(f"\nPolicy Iteration: {iterr_num} - Length of the evaluation episode: {ret_ep} (init. state: {[round(val,2) for val in starting_state]})")

    # Evaluation metric returns
    # 1. % sufficient policy counts (total sufficient policies/ total # evaluation runs)
    # 2. 'avg. episodic return'
    # 3. maximum episodic return (across all evaluations)
    # 4. minimum episodic return (across all evaluations)

    # avg_return = (sum(ep_returns)/(len(state_list)*simu_per_state))
    # pct_sr = (suf_policy_count/(len(state_list)*simu_per_state))*100

    if print_eval_summary:
        
        generate_evaluation_plot()

        print(f"\nPolicy Iteration: {iterr_num} - Evaluation results:\n \
                        - Avg. ending tumor size : {avg_t_size}\n \
                        - Avg. max. toxicity : {avg_max_toxicity}\n \
                        - Avg. prob. of death : {avg_p_death}\n")     

    return avg_t_size, avg_max_toxicity, avg_p_death 

#######################################