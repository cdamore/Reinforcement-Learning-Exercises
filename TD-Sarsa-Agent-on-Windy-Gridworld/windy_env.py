from rl_glue import BaseEnvironment
import numpy as np
import random

class WindyEnvironment(BaseEnvironment):

    def __init__(self):
        # state we are starting in
        self.startState = None

        # terminal state
        self.terminalState = 37

        # state we are in currently in
        self.currentState = None

        # current step
        self.step_count = None

        # columns of gridworld that affect state by +1 vertically
        self.single_wind = [3,4,5,8]

        # columns of gridworld that affect state by +2 vertically
        self.double_wind = [6,7]


    def env_init(self):

        # Arguments: Nothing
        # Returns: Nothing
        # init variables each run

        self.step_count = 0

    def env_start(self):

        # Arguments: Nothing
        # Returns: state - numpy array
        # start episode

        self.currentState = 30

        return self.currentState

    def env_step(self, action):

        # Arguments: action - integer
        # Returns: reward - float, state - numpy array - terminal - boolean
        # take step in enviroment

        reward = -1
        terminal = False

        # update new state following desired action
        if action == 0:
            self.currentState -= 11
        elif action == 1:
            self.currentState -= 10
        elif action == 2:
            self.currentState -= 9
        elif action == 3:
            self.currentState -= 1
        elif action == 4:
            self.currentState += 1
        elif action == 5:
            self.currentState += 9
        elif action == 6:
            self.currentState += 10
        elif action == 7:
            self.currentState += 11
        elif action != 8:
            raise Exception('action should be between 0 and 7, but was: {}'.format(action))

        # apply wind effect to new state if appliciple
        self.apply_wind()

        if self.currentState == 37:
            reward = 0
            terminal = True

        # increment step count
        self.step_count += 1

        return reward, self.currentState, terminal

    # apply wind effect to the windy gridworld
    def apply_wind(self):

        # current column the state is in (0 to 9)
        cur_column = self.currentState % 10

        # apply single wind effect if state is in single_wind column
        if (self.currentState > 10) and (cur_column in self.single_wind):
            self.currentState -= 10
        # apply single wind effect if state is in double_wind column
        elif (self.currentState > 10) and (cur_column in self.double_wind):
            self.currentState -= 10
        # apply second single wind effect if state is in double_wind column and still > 10
        if (self.currentState > 10) and (cur_column in self.double_wind):
            self.currentState -= 10

    def get_total_steps(self):
        return self.step_count
