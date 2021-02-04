import gym

######################################
### Evaluating the learned policy ####

def run_evaluations(policy               # input policy
                    , state_list         # list of initial states
                    , step_thresh = 1000    # step-count (threshold)
                    , env_name = 'CustomCartPole-v0' # name of the environment
                    , simulations_per_state = 100 # number of simulations to generate per state
                    , iterr_num = None # iterration number that the evaluation runs for
                    , print_eval_summary = None # Whether to print the evaluation summary or not
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
                observation, reward, done, _ = env_test.step(action) # execute action
                obs = observation     # set history
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
    

    # returns
    # 1. % sufficient policy counts (total sufficient policies/ total # evaluation runs)
    # 2. 'avg. episodic return'
    # 3. maximum episodic return (across all evaluations)
    # 4. minimum episodic return (across all evaluations)

    avg_return = (sum(ep_returns)/(len(state_list)*simu_per_state))

    if print_eval_summary:
        print(f"Run: {iterr_num} - Evaluation results:\n - Avg. return: {avg_return}\n - Max return: {max_return}\n - Min return: {min_return}\n")

    return (suf_policy_count/(len(state_list)*simu_per_state))*100, avg_return, max_return, min_return 

#######################################