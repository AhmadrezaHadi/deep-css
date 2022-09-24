import numpy as np
import math
import matplotlib.pyplot as plt
import gym
from gym import spaces
import parameters


class Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, pa, render=False,
                 repre='compact', end='all_done') -> None:
        super(Env, self).__init__()

        self.pa = pa
        self.action_space = spaces.Box(low=np.array([0, 0, 0]),
                                       high=np.array([self.pa.num_wq,
                                                      self.pa.num_prio, self.pa.num_serv]),
                                       dtype=np.uint8)
        if repre == 'compact':
            # TODO complete here
            self.observation_space = spaces.Box()

        self.render = render
        self.repre = repre
        self.end = end

        self.curr_time = 0

        if not self.pa.unseen:
            np.random.seed(42)

        # Initialize System
        self.machine = Machine(pa)
        self.
