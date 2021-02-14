import tqdm
import _set_path
from pbpi.algo_core.training import evaluations_per_config


# Configs to test
N_STATES = 20
configs = {'S': [N_STATES]
          , 'Actions' : [3]
          , 'Roll-outs': [10]
          , 'Significance' : [0.1]
          }


if __name__ == '__main__':

    # Original algorithm
    agg_results = []

    eval_count = len(configs['S'])*len(configs['Actions'])*len(configs['Roll-outs'])*len(configs['Significance'])

    pbar_evals = tqdm.tqdm(total=eval_count, desc="Evaluations", leave=False)

    for sample_size in configs['S']:
            
        for rollout_max in configs['Roll-outs']:

            for sig_lvl in configs['Significance']:

                run_results = evaluations_per_config(s_size = sample_size
                                                    , n_actions = configs['Actions'][0]
                                                    , max_n_rollouts = rollout_max
                                                    , sig_lvl = sig_lvl
                                                    , max_policy_iter_per_run = 2
                                                    , runs_per_config = 2
                                                    
                                                    , off_policy_explr = False # Original algo.

                                                    , rollout_tracking = False
                                                    , dataset_tracking = False
                                                    , train_plot_tracking = False
                                                    , eval_summary_tracking = True 
                                                    , show_experiment_eval_plot = True)

                agg_results.append(run_results)

                pbar_evals.update(1)
                    
    pbar_evals.close()