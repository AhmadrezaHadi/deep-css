from inspect import classify_class_attrs
from mimetypes import init
import numpy as np
import job_distribution


class Parameters:
    def __init__(self) -> None:

        self.sim_len = -1               # Simulation length
        # Maximum episode length (terminate after)
        self.episode_max_length = -1

        self.num_serv = -1              # Number of servers
        self.num_wq = -1                # Number of works in the visible queue
        self.num_prio = -1              # Number of different priorities in queue

        self.time_horizon = -1          # Number of timesteps in the graph
        self.max_job_len = -1           # Maximum duration of new job

        self.backlog_size = -1          # Backlog queue size
        self.max_track_since_new = -1   # Track how many timesteps since last new job

        self.new_job_rate = -1          # lambda in new job arrival Poisson Process
