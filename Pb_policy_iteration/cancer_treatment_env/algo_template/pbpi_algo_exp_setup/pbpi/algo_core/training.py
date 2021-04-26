import gym
import chemo_simulation  # custom chemo-simulation environment
import pandas as pd
import numpy as np
import tqdm

from pbpi.algo_core.pbpi_algo_base import evaluate_preference, train_model
from pbpi.algo_core.evaluations import run_evaluations
from pbpi.algo_core.policy import Policy
from pbpi.algo_core.create_state_pool import create_initial_state_set
import pbpi.algo_core.algo_file_paths as f_paths # path configurations

import matplotlib.pyplot as plt

import itertools

# generate a random action from a given environment
def random_action(environment, seed=10):
    """ return a random action from the given environment. """
    
    # set env. seeds for reproducibility
    #environment.action_space.np_random.seed(seed) 
    #environment.seed(seed) 
    
    return environment.action_space.sample()


def partition_action_space(env_name:'string'
                            , n_actions:'int'
                            , fixed=False):
    """function to partitions the action space of an environment into a given number of actions`"""
    
    # Initialize environment
    env = gym.make(env_name)

    if not fixed:
        # Partition the action space to a given number of actions
        part_act_space = np.linspace(env.action_space.low[0]
                                    ,env.action_space.high[0], n_actions)
    else:
        part_act_space = np.array([0.1, 0.4, 0.7, 1.0])

    return part_act_space


def evaluations_per_config(s_size
                            , n_actions
                            , max_n_rollouts
                            , sig_lvl
                            , runs_per_config = 10
                            , max_policy_iter_per_run = 10
                            , eval_runs_per_state = 100
                            , treatment_length_train = 6
                            , treatment_length_eval = 6
                            , off_policy_explr = False
                            , env_name = 'ChemoSimulation-v0'
                            , init_state_path: str = None
                            , show_experiment_run_eval_summary_plot = False
                            , rollout_tracking = False
                            , dataset_tracking = False
                            , train_plot_tracking = False
                            , eval_summary_tracking = False
                            , policy_behaviour_tracking = False
                            , set_seed = None
                            , init_state_tag = 'None'
                            , init_state_scenario = False
                            , use_toxi_n_tumor_for_pref = False
                            ):
    
    #########################
    ### PARAMETER INPUTS ###

    ## hyper-parameters ##
    env_name = env_name

    if set_seed is not None:
        this_seed = set_seed
    else:
        this_seed = np.random.randint(100) #51

    # Load custom initial state data if provided
    if init_state_path is not None:
        INIT_STATES = pd.read_csv(init_state_path)
    else:
        INIT_STATES = create_initial_state_set(s_size, seed = this_seed, init_state_scenario = init_state_scenario)
    
    print(f"\nState generation seed is {this_seed}\n")

    NUM_SAMPLES = len(INIT_STATES)

    s_size = s_size             # initial state stample size
    n_actions = n_actions       # number of actions in the action space
    n_rollouts = max_n_rollouts # max. number of roll-outs to generate per action
    sig_lvl = sig_lvl           # statistical significance for action-pair comparisons
    runs_per_config = runs_per_config  # training runs for a single parameter configuration

    # hyper-parameter configurations (string)
    param_config_string = f'Samples: {s_size} | Actions: {n_actions} | Roll-outs: {n_rollouts} | Significance: {sig_lvl}'
    
    ## task settings ##

    seed = 2                                  # set seed
    max_iterr = max_policy_iter_per_run       # max. num. of policy iterations
    off_policy_exploration = off_policy_explr # trigger to use off-policy exploration [MY MODIFICATION]
    eval_simu_per_state = eval_runs_per_state # number of evaluation runs from each initial starting state (evaluation)
    
    method_name = 'Mod_algo' if off_policy_explr else 'Orig_algo'                  # string to store whether modified/original algo is running
    model_name = f'{method_name}_Chemo_{n_rollouts}_{sig_lvl}_state_tag_{init_state_tag}'      # name for the saved LabelRanker model

    ## flags/triggers ##

    print_iterr = False                   # trigger to print progress bars of training iterations

    ###############################

    ### Variable initialization ###

    sample_states = np.array(INIT_STATES).reshape(NUM_SAMPLES,2)  # generate sample states
    act_space = partition_action_space(env_name = env_name, n_actions = n_actions, fixed=True) # partition the action space
    act_pairs = list(itertools.combinations(act_space,2)) # generate action-pairs from the partitioned action space

    print(f'\nCurrently evaluated configs:\n '+  param_config_string)

    # Initialize the LabelRanker model and epoch configs
    # Note: these configs were decided after testing different settings; there can be better/different choices
    if s_size < 10000:
        model_config = [50]
        epch_config  = 2000
        l_rate_config = 0.001
        batch_s_config = 10
    # elif s_size >= 49 and s_size < 149:
    #     model_config = [100]
    #     epch_config  = 2000
    #     l_rate_config = 0.001
    #     batch_s_config = 5
    # else:
    #     model_config  = [125]
    #     epch_config   = 2000
    #     l_rate_config = 0.001
    #     batch_s_config = 5


    # list to store results of the evaluation run
    run_results = []

    # generate evaluations for a single hyper-parameter configuration
    for run in tqdm.tqdm(range(runs_per_config), desc="Runs"):

        ### place holders for evaluation metrics ###

        # lists to store the evaluation metrics of the run
        avg_tsize_welness_at_end_l = []
        #avg_max_tox_l= []
        avg_prob_death_l= []

        action_count_li = []       # list to store the action counts in each training iteration


        ### flags, triggers and adjustments ###

        label_r_flag = False       # trigger to start using the trained LabelRanker model 
        policy = random_action     # set the initial policy to a random policy
        _max_iterr = max_iterr + 1  # since iteration count starts from '1', increment the max. iteration count by 1


        ### training loop ###

        iterr = 1
        while iterr < _max_iterr:

            train_data = []      # place-holder to store training data
            actions_in_iterr = 0 # variable to store the num. actions excuted in each training iteration

            for state in sample_states: # generate roll-outs from each starting state

                for action_pair in act_pairs: # generate roll-outs for each action pair

                    # generate preference data & executed num. of actions in each action pair evaluation step
                    preference_out, actions_per_pair = evaluate_preference(starting_state = state
                                                                            , action_1       = action_pair[0]
                                                                            , action_2       = action_pair[1]
                                                                            , policy_in      = policy
                                                                            , label_ranker   = label_r_flag
                                                                            , modified_algo  = True if off_policy_exploration else False
                                                                            , n_rollouts     = n_rollouts
                                                                            , p_sig          = sig_lvl
                                                                            , tracking       = rollout_tracking
                                                                            , max_rollout_len = treatment_length_train
                                                                            , use_toxi_n_tsize = use_toxi_n_tumor_for_pref
                                                                            )   

                    # append the generated preference data to the training data list
                    if preference_out is not None:
                        train_data.append(preference_out) 
                    else:
                        pass

                    # compute/update the tot. # actions executed in the training iteration
                    actions_in_iterr += actions_per_pair  

            # generate the training dataset and learn the LabelRanker model
            model = train_model(train_data     = train_data
                                , action_space = act_space
                                , model_name   = model_name 
                                , mod_layers   = model_config
                                , batch_s      = batch_s_config
                                , n_epochs     = epch_config 
                                , l_rate       = l_rate_config
                                , retrain_model = True if off_policy_exploration else False
                                , policy_iterr_count = iterr
                                , show_train_plot = train_plot_tracking
                                , show_dataset    = dataset_tracking
                                )


            # When no traiing data is found, the LabelRanker model will not be trained. 
            # Therefore, break the current training iteration and continue to the next 
            # (after updating the aggregated evaluation results)
            if model is None:

                print(f'No training data collected!')

                # update the tot. # actions executed across all training iterations
                if iterr>1:
                    action_count_li.append(actions_in_iterr+action_count_li[iterr-2])
                else:
                    action_count_li.append(actions_in_iterr)

                # Add None to the evaluation results
                avg_tsize_welness_at_end_l.append(None)
                #avg_max_tox_l.append(None)
                avg_prob_death_l.append(None)

                iterr += 1
                continue


            # Derive a new policy using the trained model
            if off_policy_exploration:
                # Generate separate 'target' and 'behaviour' policies
                # Target policy to be used in evaluations, and behaviour policy to generate roll-outs (training data)
                
                # If there are more than two actions, assign zero probability to remaining actions
                if len(act_space)>2:
                    prob_fill =  np.repeat(0,len(act_space)-2)
                    
                target_policy = Policy(act_space, model, [1.0, 0.0]+list(prob_fill), modified_algo_flag = True) # always select the highest ranked action
                exp_policy = Policy(act_space, model, [0.5, 0.5]+list(prob_fill), modified_algo_flag = True)    # select the first two highest ranked actions w/ same prob. 

            else:
                # Set both 'target' and 'behaviour' policies to follow the optimal policy
                # I.e., always select the highest ranked action
                
                # If there are more than two actions, assign zero probability to remaining actions
                if len(act_space)>2:
                    prob_fill =  np.repeat(0,len(act_space)-2)

                target_policy = Policy(act_space, model, [1.0, 0.0]+list(prob_fill))
                exp_policy = Policy(act_space, model, [1.0, 0.0]+list(prob_fill))


            # update the tot. # actions executed across all training iterations
            if iterr>1:
                action_count_li.append(actions_in_iterr+action_count_li[iterr-2])
            else:
                action_count_li.append(actions_in_iterr)


            # evaluate the performance of the learned policy
            avg_tsize_welness_at_end, avg_prob_death = run_evaluations(target_policy
                                                                    #, sample_states
                                                                    , simulations_per_state = eval_simu_per_state
                                                                    , virtual_patients = 200 
                                                                    , sim_episode_length = treatment_length_eval
                                                                    , iterr_num = iterr
                                                                    , print_eval_summary = eval_summary_tracking
                                                                    , print_policy_behaviour = policy_behaviour_tracking
                                                                    , model_name_input =  model_name
                                                                    , experiment_run_input = run+1
                                                                    , init_state_scenario = init_state_scenario
                                                                    , set_seed_eval = this_seed
                                                                    , init_state_tag = init_state_tag
                                                                    ) 


            # record evaluation results (across training iterations)
            avg_tsize_welness_at_end_l.append(avg_tsize_welness_at_end)
            #avg_max_tox_l.append(avg_max_tox)
            avg_prob_death_l.append(avg_prob_death)

            ### TERMINATION CONDITION ###

            # If the current policy's performance (% of sufficient policies) is less than 
            #  half of the last policy's performance, TERMINATE the training process

            if iterr>1:
                prvs_avg_prob_death = avg_prob_death_l[-2]
                curr_avg_prob_death = avg_prob_death_l[-1]

                prvs_avg_tsize_welness = avg_tsize_welness_at_end_l[-2]
                curr_avg_tsize_welness = avg_tsize_welness_at_end_l[-1]

                # Policy iteration Termination criteria
                if prvs_avg_prob_death * (1.1) <= curr_avg_prob_death:
                    print(f'Averege death rate increased by 10%! Policy performance decreased! Run-{run+1} terminated!')
                    # remove the records from the worsen policy
                    avg_prob_death_l = avg_prob_death_l[:-1]
                    #avg_max_tox_l = avg_max_tox_l[:-1]
                    avg_tsize_welness_at_end_l = avg_tsize_welness_at_end_l[:-1]
                    action_count_li = action_count_li[:-1]                    
                    break

                elif prvs_avg_tsize_welness * (1.1) <= curr_avg_tsize_welness:
                    print(f'Averege tumor-size + toxicity increased by 10%! Policy performance decreased! Run-{run+1} terminated!')
                    # remove the records from the worsen policy
                    avg_prob_death_l = avg_prob_death_l[:-1]
                    #avg_max_tox_l = avg_max_tox_l[:-1]
                    avg_tsize_welness_at_end_l = avg_tsize_welness_at_end_l[:-1]
                    action_count_li = action_count_li[:-1]                    
                    break

                elif (prvs_avg_tsize_welness * (1.05) <= curr_avg_tsize_welness) and (prvs_avg_prob_death * (1.05) <= curr_avg_prob_death):
                    print(f'Both averege death rate and tumor-size + toxicity increased by 5%! Policy performance decreased! Run-{run+1} terminated!')
                    # remove the records from the worsen policy
                    avg_prob_death_l = avg_prob_death_l[:-1]
                    #avg_max_tox_l = avg_max_tox_l[:-1]
                    avg_tsize_welness_at_end_l = avg_tsize_welness_at_end_l[:-1]
                    action_count_li = action_count_li[:-1]                    
                    break

                    
            # Start using the trained LabelRanker model
            # The first policy of the training process is always a random-policy
            # From the second iteration onward, it uses the learned LabelRanker model
            label_r_flag = True

            if label_r_flag is False:
                policy = random_action # set the random policy
            else:
                policy = exp_policy

            iterr += 1

        # plot and save evaluation results of the training run 
        fig, ax = plt.subplots(figsize = (12,8))
        ax.plot(action_count_li, avg_prob_death_l, 'r--', label = 'probability of death')
        ax.set_xlabel('# actions', fontsize=12)
        ax.set_ylabel('Probability of Death', fontsize=12)

        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()
        ax2.plot(action_count_li, avg_tsize_welness_at_end_l, 'b--', label = 'tumor size+ toxicity')
        ax2.set_ylabel('average tumor size+ toxicity', fontsize=12 )

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title(f'Experiment Evaluation Results | Run: {run+1}\n', fontsize=14)
        plt.savefig(f_paths.paths['eval_plot_output'] + f'{model_name}_{run+1}.png') # save the evaluation image

        if show_experiment_run_eval_summary_plot: 
            plt.show() 
        
        # store the evaluation results of the training run
        run_results.append({'S': s_size
                            , 'Actions' : n_actions
                            , 'Roll-outs': n_rollouts
                            , 'Significance' : sig_lvl
                            , 'run': run+1
                            , 'action_record': action_count_li
                            #, 'avg_tumor_size': avg_t_size_l
                            #, 'avg_max_toxicity' : avg_max_tox_l
                            , 'avg_max_toxicity' : avg_tsize_welness_at_end_l
                            , 'avg_prop_death': avg_prob_death_l
                            })

        if print_iterr:
            #pbar.close()
            pass
            
    # output the recorded evaluation results for the hyper-parameter configuration
    return run_results