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
                    , adjust_tumor = False
                    , set_seed_eval = None
                   ):  
                   
    """
    Description:
    
        - For every state in a given list of initial states, 100 simulations are generate and the percentage of
        these simulations that exceeds a predefined step-count threadhold (trajectory length) is computed to measure 
        the performance of the given input policy."""
    

    simu_per_state = simulations_per_state

    #print(f"\nEvaluation 200 patient generation seed is: {set_seed_eval}\n")

    # Create 200 virtual patients | new set of patients generated in each evaluation
    INIT_STATES = create_initial_state_set(virtual_patients, seed = set_seed_eval, adjust_tumor =  adjust_tumor)

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
    
    state_count = 0
    state_data_list = []

    state_list = INIT_STATES
    # iterate over all states in the state list
    for state in state_list:        
        
        state_count += 1
        
        # generate simulations from each state
        for _ in range(simu_per_state):
            
            # set the starting state and the current observation to the given state 
            env_test.reset(init_state=state)
            obs = state

            state_data_dict = {}
            state_data_dict = { 'treatment' : 'Learned'
                                ,'patient_no' : state_count
                                , 'tumor_size' : state[0]
                                , 'negative_wellness' : state[1]
                                , 'month': 0
                                }

            state_data_list.append(state_data_dict)

            # # variable to store the return of an episode
            # return_ep = 0 

            # execute steps in the environment
            for i in range(sim_episode_length):

                action = policy.label_ranking_policy(obs) # generate action from the policy
                obs, reward, done, p_death = env_test.step(action) # execute action
                #obs = observation     # set history
                #return_ep += reward   # compute return

                state_data_dict = {}
                state_data_dict = {'treatment' : 'Learned'
                                    , 'patient_no' : state_count
                                    , 'tumor_size' : obs[0][0]
                                    , 'negative_wellness' : obs[1][0]
                                    , 'month': i+1
                                    }

                state_data_list.append(state_data_dict)

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

    lrned_p_metrics_df = pd.DataFrame(learned_policy_dic, index=['Learned']) 

    learned_policy_state_data_df = pd.DataFrame(state_data_list)

    plt.close('all')
    
    def generate_evaluation_plot(init_state_set=INIT_STATES, learned_policy_metrics_df = lrned_p_metrics_df):

        # Create instance of the environment
        env = gym.make('ChemoSimulation-v0')

        # Dosage levels to test
        dosages =    {'Low': np.array([0.1])
                    , 'Mid': np.array([0.4])
                    , 'Random': None
                    , 'High': np.array([0.7])
                    , 'Extreme': np.array([1.])}


        # Select the initial set of patients
        selected_patients = init_state_set

        # Dict to store patient information
        patient_dic = {}

        patient_count = 0

        p_data_list = []

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

                p_data_dict = {}
                p_data_dict['patient_no'] = patient_count
                p_data_dict['month'] = 0
                p_data_dict['tumor_size'] = patient[0]
                p_data_dict['negative_wellness'] = patient[1]
                p_data_dict['treatment'] = level
                p_data_list.append(p_data_dict)

                # Simulate the environment
                env.reset(init_state = starting_state)

                if level == 'Random':
                    # Run treatment for 6 months
                    for i in range(sim_episode_length):
                        obs, reward, done, p_death = env.step(np.array([np.random.choice(np.array([0.1,0.4,0.7,1.]))]))
                    
                        obs_dic[level].append(obs)
                        p_death_dic[level] = p_death

                        p_data_dict = {}
                        p_data_dict['patient_no'] = patient_count
                        p_data_dict['month'] = i+1
                        p_data_dict['tumor_size'] = obs[0][0]
                        p_data_dict['negative_wellness'] = obs[1][0]
                        p_data_dict['treatment'] = level
                        p_data_list.append(p_data_dict)
                        
                        if done:
                            break

                else:
                    # Run treatment for 6 months
                    for i in range(sim_episode_length):
                        obs, reward, done, p_death = env.step(dosage)
                        
                        obs_dic[level].append(obs)
                        p_death_dic[level] = p_death

                        p_data_dict = {}
                        p_data_dict['patient_no'] = patient_count
                        p_data_dict['month'] = i+1
                        p_data_dict['tumor_size'] = obs[0][0]
                        p_data_dict['negative_wellness'] = obs[1][0]
                        p_data_dict['treatment'] = level
                        p_data_list.append(p_data_dict)
                        
                        if done:
                            break

                env.close()

        patient_dic[patient_count] = obs_dic, p_death_dic
        patient_count += 1
        
        constant_policy_state_data_df = pd.DataFrame(p_data_list)
        all_policy_state_data_df = constant_policy_state_data_df.append(learned_policy_state_data_df)
        #all_policy_state_data_df.to_csv('delete_this_data2.csv', index=False)

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

        # Join learned policy performance
        plot_df_summary = pd.concat([plot_df_summary, learned_policy_metrics_df])  


        ##################### Plotting data #####################

        # Data for the connecting line (excluding random dosage)
        line_y = plot_df_summary.loc[(plot_df_summary.index == 'Extreme') \
                                    | (plot_df_summary.index == 'High') \
                                    | (plot_df_summary.index == 'Mid') \
                                    | (plot_df_summary.index == 'Low'),'max_toxicity'].values

        line_x = plot_df_summary.loc[(plot_df_summary.index == 'Extreme') \
                                    | (plot_df_summary.index == 'High') \
                                    | (plot_df_summary.index == 'Mid') \
                                    | (plot_df_summary.index == 'Low') , 'end_tumor_size'].values     

        # Create the plot
        fig, ax = plt.subplots(nrows =  1, ncols=2, figsize = (14,5))

        sns.scatterplot(x='end_tumor_size'
                    , y = 'max_toxicity'
                    , data = plot_df_summary
                    , s=120
                    , ax = ax[0]
                    )

        for i in range(plot_df_summary.shape[0]):
            ax[0].text(x = plot_df_summary.end_tumor_size[i]+0.2
                        , y = plot_df_summary.max_toxicity[i]+0.2
                        , s = plot_df_summary.index[i]
                        , fontdict = dict(color='black', size=11)
                    )

        sns.lineplot(x = line_x
                    , y = line_y
                    , ax = ax[0])

        ax[0].lines[0].set_linestyle("--")
        ax[0].set(ylim=(0, plot_df_summary.max_toxicity.max() + .5))
        ax[0].grid()
        ax[0].set_title ('Max Toxicity vs. Tumor size')
        ax[0].set_ylabel('Max toxicity')
        ax[0].set_xlabel('Tumor size')

        sns.barplot(y = 'prob_death'
                    , x = plot_df_summary.index
                    , data = plot_df_summary
                    , ax =  ax[1]
                    , palette = sns.color_palette("mako"))


        ax[1].set_title ('Average probability of death by treatment-type')
        ax[1].set_ylabel('Probability of death')
        ax[1].set_xlabel('Treatment type')
        plt.savefig(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_behaviour.png') # save the evaluation image
        plt.show()  


        ### Generate treatment regime plots ###
        all_policy_state_data_df.loc[:, 'tumor_size_welness'] = all_policy_state_data_df.tumor_size + all_policy_state_data_df.negative_wellness
        plot_df = all_policy_state_data_df.groupby(['treatment','month']).mean().reset_index()

        x_col = 'month'
        hue_col = "treatment"
        style_col = "treatment"

        #color palette
        cmap = sns.color_palette("bright")
        palette = {key:value for key,value in zip(plot_df[hue_col].unique(), cmap)}
        palette['Learned'] = 'red'
        palette['Random'] = 'blue'
        palette['Low'] = 'darkgreen'
        palette['Mid'] = 'darkorange'
        palette['High'] = 'magenta'
        palette['Extreme'] = 'sienna'

        #style palette
        #dash_list = sns._core.unique_dashes(data[style_col].unique().size+1)
        dash_list = ['', (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
        style = {key:value for key,value in zip(plot_df[style_col].unique(), dash_list[1:])}
        style['Learned'] = ''  # empty string means solid

        #sns.set_theme()
        sns.set_style("ticks")
        #sns.set_context("paper")

        fig, ax = plt.subplots(figsize=(10,15),nrows=3,ncols=1)
        sns.lineplot(data=plot_df
                    , x=x_col
                    , y='tumor_size'
                    , hue=hue_col
                    , palette=palette
                    , style=style_col
                    , dashes=style
                    , linewidth =2
                    , ci = None
                    , ax=ax[0])

        ax[0].set_xlabel('Month', fontsize = 13)
        ax[0].set_ylabel('Negative wellness\n', fontsize = 13)
        ax[0].set_title('Tumor size: Averaged across 200 patients', fontsize=14)
        ax[0].legend(bbox_to_anchor=(1.02, 1),borderaxespad=0, fontsize=12)

        sns.lineplot(data=plot_df
                    , x=x_col
                    , y='negative_wellness'
                    , hue=hue_col
                    , palette=palette
                    , style=style_col
                    , dashes=style
                    , linewidth =2
                    , ci = None
                    , ax=ax[1])

        ax[1].set_xlabel('Month', fontsize = 13)
        ax[1].set_ylabel('Negative wellness\n', fontsize = 13)
        ax[1].set_title('(Negative) wellness: Averaged across 200 patients', fontsize=14)
        ax[1].legend(bbox_to_anchor=(1.02, 1),borderaxespad=0, fontsize=12)

        sns.lineplot(data=plot_df
                    , x=x_col
                    , y='tumor_size_welness'
                    , hue=hue_col
                    , palette=palette
                    , style=style_col
                    , dashes=style
                    , linewidth =2
                    , ci = None
                    , ax=ax[2])

        ax[2].set_xlabel('Month', fontsize = 13)
        ax[2].set_ylabel('Tumor size + Negative wellness\n', fontsize = 13)
        ax[2].set_title('Tumor Size + (Negative) wellness: Averaged across 200 patients', fontsize=14)
        ax[2].legend(bbox_to_anchor=(1.02, 1),borderaxespad=0, fontsize=12)

        fig.tight_layout()
        fig.savefig(f_paths.paths['policy_behavior_output'] + f'{model_name_input}_run_{experiment_run_input}_iterr_{iterr_num}_policy_bhve_monthly.png') # save the evaluation image
        plt.show()

    if print_eval_summary:
        
        generate_evaluation_plot()

        print(f"\nPolicy Iteration: {iterr_num} - Evaluation results:\n \
                        - Avg. ending tumor size : {avg_t_size[0]}\n \
                        - Avg. max. toxicity : {avg_max_toxicity[0]}\n \
                        - Avg. prob. of death : {avg_p_death[0]}\n")     

    return avg_t_size, avg_max_toxicity, avg_p_death 

#######################################