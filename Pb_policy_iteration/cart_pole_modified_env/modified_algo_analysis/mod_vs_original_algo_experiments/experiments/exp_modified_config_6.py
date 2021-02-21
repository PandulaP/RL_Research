import tqdm
import pandas as pd
import _set_path
from pbpi.algo_core.training import evaluations_per_config

ALGO_TYPE = {'original': {'name': 'original', 'exploration': False}
            ,'modified': {'name': 'modified', 'exploration': True} }

############################################
################## INPUTS ##################

# Configs to test
N_STATES = 50

configs = { 'CONFIG_NO': 6
          , 'S': [N_STATES]
          , 'Actions' : [3]
          , 'Roll-outs': [50]
          , 'Significance' : [0.1, 0.05, 0.025]
          }

algorithm = ALGO_TYPE['modified']

############################################
############################################

# Algorithm configs
ALGO_NAME = algorithm['name']
EXPLORE_LOGIC = algorithm['exploration']

if __name__ == '__main__':

    # Original algorithm
    agg_results = []

    eval_count = len(configs['S'])*len(configs['Actions'])*len(configs['Roll-outs'])*len(configs['Significance'])

    pbar_evals = tqdm.tqdm(total=eval_count, desc="Evaluations")

    for sample_size in configs['S']:
            
        for rollout_max in configs['Roll-outs']:

            for sig_lvl in configs['Significance']:

                run_results = evaluations_per_config(s_size          = sample_size
                                                    , n_actions      = configs['Actions'][0]
                                                    , max_n_rollouts = rollout_max
                                                    , sig_lvl        = sig_lvl

                                                    , max_policy_iter_per_run = 10
                                                    , runs_per_config         = 10
                                                    
                                                    , off_policy_explr = EXPLORE_LOGIC

                                                    , rollout_tracking          = False
                                                    , dataset_tracking          = False
                                                    , train_plot_tracking       = False
                                                    , eval_summary_tracking     = False 
                                                    , show_experiment_eval_plot = False
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