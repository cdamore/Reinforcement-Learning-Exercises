from rl_glue import RLGlue
from windy_env import WindyEnvironment
from sarsa_agent import SarsaAgent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_episodes = 170
    max_steps = 10000
    num_runs = 2
    # Create and pass agent and environment objects to RLGlue
    environment = WindyEnvironment()
    agent = SarsaAgent()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore
    # episodes at which we are interested in the value function
    key_episodes = [99, 999, 7999]
    value_functions = [] # dict to hold data for key episodes
    v_over_runs = {}

    for episode in key_episodes:
        v_over_runs[episode] = []

    for run in range(num_runs):
        # set seed for reproducibility
        np.random.seed(run)
        # initialize RL-Glue
        rlglue.rl_init()
        total_steps = []
        if run == 0:
            rlglue._agent.set_num_of_actions(8)
            print("running sarsa agent with 8 actions...")
        else:
            rlglue._agent.set_num_of_actions(9)
            print("running sarsa agent with 9 actions...")

        # loop over episodes
        for episode in range(num_episodes):
            # run episode with the allocated steps budget
            rlglue.rl_episode(max_steps)
            total_steps.append(rlglue._environment.get_total_steps())

        # plot
        p1, = plt.plot(total_steps, range(1,171))
        plt.ylabel('Episodes')
        plt.xlabel('Time Steps')
        if run == 0:
            plt.title('Performace of Sarsa Agent with 8 Actions')
        else:
            plt.title('Performace of Sarsa Agent with 9 Actions')
        plt.show()
