import tqdm
import pandas as pd
import _set_path
from pbpi.algo_core.training import evaluations_per_config

ALGO_TYPE = {'original': {'name': 'original', 'exploration': False}
            ,'modified': {'name': 'modified', 'exploration': True} }

############################################
################## INPUTS ##################

# Configs to test
N_STATES = 40

configs = { 'CONFIG_NO': 1
          , 'S': [N_STATES]
          , 'Actions' : [3]
          , 'Roll-outs': [10, 20, 50, 100]
          , 'Significance' : [0.1, 0.05, 0.025]
          , 'init_state_path': './manual_init_state_input/uniformly_sampled_states.csv'
          }

algorithm = ALGO_TYPE['original']

############################################
############################################

# Algorithm configs
ALGO_NAME = algorithm['name']
EXPLORE_LOGIC = algorithm['exploration']

if __name__ == '__main__':

    agg_results = []

    eval_count = len(configs['S'])*len(configs['Actions'])*len(configs['Roll-outs'])*len(configs['Significance'])

    pbar_evals = tqdm.tqdm(total=eval_count, desc="Evaluations")

    for sample_size in configs['S']:
            
        for rollout_max in configs['Roll-outs']:

            for sig_lvl in configs['Significance']:

                run_results = evaluations_per_config(s_size          = sample_size
                                                    #, init_state_path       = configs['init_state_path'] # Use a pre-designed init state configs   
                                                    , n_actions      = configs['Actions'][0]
                                                    , max_n_rollouts = rollout_max
                                                    , sig_lvl        = sig_lvl

                                                    , max_policy_iter_per_run = 10 # Maximum number of policy iterations per experiment
                                                    , runs_per_config         = 10 # Number of experiments per one parameter config

                                                    , eval_runs_per_state     = 100 # Episodes to generate from each init. state (during evaluation)
                                                    
                                                    , off_policy_explr = EXPLORE_LOGIC # What algorithm to use

                                                    , rollout_tracking          = False # Show rollout info.
                                                    , dataset_tracking          = False # Show train dataset

                                                    , train_plot_tracking       = False # Show model training plot
                                                    , eval_summary_tracking     = False # Show a policy performance summary of evaluation runs
                                                    , policy_behaviour_tracking = False # Show/store policy action selections vs. pendulum angle plot

                                                    , show_experiment_run_eval_summary_plot = False # Show SR vs. action no. plot of exp. run
                                                    )

                agg_results.append(run_results)

                pbar_evals.update(1)
                    
    pbar_evals.close()

    # Save the evaluation results
    results_dfs = []
    for result in agg_results:
        results_dfs.append(pd.DataFrame(result))

    results_df = pd.concat(results_dfs)

    results_df.to_excel(f"eval_results/{ALGO_NAME}_experiment_results_para_config_{configs['CONFIG_NO']}.xlsx", index=False)