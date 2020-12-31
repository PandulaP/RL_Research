from gym.envs.registration import register

register(id = 'ChemoSimulation-v0'
         , entry_point = 'chemo_simulation.envs:ChemoSimulationEnv'
         , max_episode_steps = 1500
)
