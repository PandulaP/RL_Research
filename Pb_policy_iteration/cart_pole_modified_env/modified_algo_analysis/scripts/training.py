import gym
import custom_cartpole  # custom cart-pole environment
import pandas as pd
import numpy as np
import tqdm

from pbpi_algo import evaluate_preference, train_model
from evaluations import run_evaluations
from policy import Policy

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
                           , n_actions:'int'):
    """function to partitions the action space of an environment into a given number of actions`"""
    
    # Initialize environment
    env = gym.make(env_name)

    # Partition the action space to a given number of actions
    part_act_space = np.linspace(env.action_space.low[0,0]
                                 ,env.action_space.high[0,0],n_actions)
    
    return part_act_space


def evaluations_per_config(s_size 
                           , n_actions
                           , max_n_rollouts
                           , sig_lvl
                           , runs_per_config = 10
                           , max_policy_iter_per_run = 10
                           , off_policy_explr = False
                           , env_name = 'CustomCartPole-v0'
                           , init_state_path: str = None
                           , print_run_eval_plot = False
                           , rollout_tracking = False
                           , dataset_tracking = False
                           , train_plot_tracking = False
                           , eval_summary_tracking = False
                           ):
    
    #########################
    ### PARAMETER INPUTS ###

    ## hyper-parameters ##
    env_name = env_name

    INIT_STATES = pd.read_csv(init_state_path)
    NUM_SAMPLES = INIT_STATES.shape[0]

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
    eval_simu_per_state = 100                 # number of evaluation runs from each initial starting state (evaluation)
    
    model_name = f'CartPole_{s_size}_{n_actions}_{n_rollouts}_{sig_lvl}'      # name for the saved LabelRanker model

    ## flags/triggers ##

    print_iterr = False                   # trigger to print progress bars of training iterations

    #########################

    ### variable initialization ###

    #env = gym.make(env_name)   # create environment
    sample_states = INIT_STATES.values.reshape(NUM_SAMPLES,4,1,1)  # generate sample states
    act_space = partition_action_space(env_name = env_name, n_actions = n_actions) # partition the action space
    act_pairs = list(itertools.combinations(act_space,2)) # generate action-pairs from the partitioned action space

    print(f'\nCurrently evaluated configs:\n '+  param_config_string, end='\r')

    # Initialize the LabelRanker model and epoch configs
    # Note: these configs were decided after testing different settings; there can be better/different choices
    if s_size < 49:
        model_config = [20]
        epch_config  = 1000
        l_rate_config = 0.001
        batch_s_config = 5
    elif s_size >= 49 and s_size < 149:
        model_config = [100]
        epch_config  = 2000
        l_rate_config = 0.001
        batch_s_config = 5
    else:
        model_config  = [125]
        epch_config   = 2000
        l_rate_config = 0.001
        batch_s_config = 5


    # list to store results of the evaluation run
    run_results = []

    # generate evaluations for a single hyper-parameter configuration
    for run in tqdm.tqdm(range(runs_per_config), desc="Runs"):

        ### place holders for evaluation metrics ###

        agg_pct_suff_policies = [] # list to store the % of learned sufficient policies
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
                                                                         , action_1       = np.array([[action_pair[0]]])
                                                                         , action_2       = np.array([[action_pair[1]]])
                                                                         , policy_in      = policy
                                                                         , label_ranker   = label_r_flag
                                                                         , modified_algo  = True if off_policy_exploration else False
                                                                         , n_rollouts     = n_rollouts
                                                                         , p_sig          = sig_lvl
                                                                         , tracking       = rollout_tracking
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

                # Add '0' to the evaluation results
                agg_pct_suff_policies.append(0) # pct. of sufficient policies in evaluations

                iterr += 1
                continue


            # Derive a new policy using the trained model
            if off_policy_exploration:

                # Generate separate 'target' and 'behaviour' policies
                # Target policy to be used in evaluations, and behaviour policy to generate roll-outs (training data)
                target_policy = Policy(act_space, model, [1.0, 0.0, 0.0], modified_algo_flag = True) # always select the highest ranked action
                exp_policy = Policy(act_space, model, [0.5, 0.5, 0.0], modified_algo_flag = True)    # select the first two highest ranked actions w/ same prob. 

            else:

                # Set both 'target' and 'behaviour' policies to follow the optimal policy
                # I.e., always select the highest ranked action
                target_policy = Policy(act_space, model, [1.0, 0.0, 0.0])
                exp_policy = Policy(act_space, model, [1.0, 0.0, 0.0])


            # update the tot. # actions executed across all training iterations
            if iterr>1:
                action_count_li.append(actions_in_iterr+action_count_li[iterr-2])
            else:
                action_count_li.append(actions_in_iterr)


            # evaluate the performance of the learned policy
            pct_succ_policies, _, _, _ = run_evaluations(target_policy
                                                        , sample_states
                                                        , simulations_per_state = eval_simu_per_state
                                                        , step_thresh = 1000 # steps needed for a sufficient policy
                                                        , iterr_num = iterr
                                                        , print_eval_summary = eval_summary_tracking
                                                       ) 


            # record evaluation results (across training iterations)
            agg_pct_suff_policies.append(pct_succ_policies) # pct. of sufficient policies in evaluations


            ### TERMINATION CONDITION ###

            # If the current policy's performance (% of sufficient policies) is less than 
            #  half of the last policy's performance, TERMINATE the training process

            if iterr>1:
                prvs_policy_perf = agg_pct_suff_policies[-2]
                curr_policy_perf = agg_pct_suff_policies[-1]

                if prvs_policy_perf * (0.5) > curr_policy_perf:
                    print(f'Policy performance decreased! Run-{run} terminated!')

                    # remove the records from the worsen policy
                    agg_pct_suff_policies = agg_pct_suff_policies[:-1]
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

        # plot evaluation results of the training run 
        if print_run_eval_plot: 
            
            plt.clf()
            plt.cla()
            plt.close()

            fig, ax2 = plt.subplots(figsize =(6,4))
            ax2.plot(action_count_li, agg_pct_suff_policies, 'm-.', label = 'success rate')
            ax2.set_xlabel('# actions')
            ax2.set_ylabel('Pct. of sufficient policies')
            ax2.legend(loc='upper left')
            plt.title(f'Evaluation Results | Run: {run+1}')

            plt.savefig(f'../data/output/train_imgs/{model_name}_{run}.png') # save the evaluation image
            plt.show() 
        
        # store the evaluation results of the training run
        run_results.append({'S': s_size
                           , 'Actions' : n_actions
                           , 'Roll-outs': n_rollouts
                           , 'Significance' : sig_lvl
                           , 'run': run
                           , 'action_record': action_count_li
                           , 'SR': agg_pct_suff_policies})

        if print_iterr:
            #pbar.close()
            pass
            
    # output the recorded evaluation results for the hyper-parameter configuration
    return run_results