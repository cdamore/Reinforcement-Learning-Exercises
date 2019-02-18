from rl_glue import BaseAgent
import numpy as np
import random

# Sarsa agent
class SarsaAgent(BaseAgent):

    def __init__(self):
        # e-greedy probability
        self.e = 0.1

        # step size for updating action-value function
        self.a = 0.5

        # number of actions for a state (4-9)
        self.num_of_actions = None

        # previous state
        self.previous_state = None

        # previous action
        self.previous_action = None

        # Agents action_value funtion
        self.action_values = None

    def agent_init(self):

        # Arguments: Nothing
        # Returns: Nothing

        if self.num_of_actions is not None:

            # Agents action_value funtion
            #self.action_values = np.random.rand(70,self.num_of_actions) - 1
            self.action_values = np.zeros((70,self.num_of_actions))

            # reduce possible actions
            for state in range(len(self.action_values)):
                if state % 10 == 0:
                    self.action_values[state][0] = None
                    self.action_values[state][3] = None
                    self.action_values[state][5] = None
                elif state % 10 == 9:
                    self.action_values[state][2] = None
                    self.action_values[state][4] = None
                    self.action_values[state][7] = None
                if state < 10:
                    self.action_values[state][0] = None
                    self.action_values[state][1] = None
                    self.action_values[state][2] = None
                elif state >= 60:
                    self.action_values[state][5] = None
                    self.action_values[state][6] = None
                    self.action_values[state][7] = None

    def agent_start(self, state):

        # Arguments: state - numpy array
        # Returns: action - integer

        # Explore or exploit
        explore_prob = random.uniform(0, 1)
        if explore_prob < self.e:
            # choose random action
            action = self.get_random_action(state)
        else:
            # greedy choice of action
            action = np.nanargmax(self.action_values[state])

        self.previous_state = state
        self.previous_action = action

        return action

    def agent_step(self, reward, state):

        # Arguments: reward - floting point, state - numpy array
        # Returns: action - integer

        # Explore or exploit
        explore_prob = random.uniform(0, 1)
        if explore_prob < self.e:
            # choose random action
            action = self.get_random_action(state)
        else:
            # greedy choice of action
            action = np.nanargmax(self.action_values[state])

        # Update
        self.action_values[self.previous_state][self.previous_action] += self.a * (reward + self.action_values[state][action] - self.action_values[self.previous_state][self.previous_action])

        #print(self.action_values[self.previous_state][self.previous_action])

        self.previous_state = state
        self.previous_action = action

        return action

    def agent_end(self, reward):
        # Arguments: reward - floating point
        # Returns: Nothing
        # no end action needed for this agent
        pass

    # gets random action for a given state
    def get_random_action(self, state):
        actions = []
        for action in range(self.num_of_actions):
            if ~np.isnan(self.action_values[state][action]):
                actions.append(action)
        return random.choice(actions)


    # sets number of possible actions for each state (4-9)
    def set_num_of_actions(self, num):
        self.num_of_actions = num
        self.agent_init()
