import numpy as np
from . import job_distribution


class Parameters:
    def __init__(self) -> None:

        # Simulation length i.e. number of incoming jobs at each simulation
        self.simu_len = 30
        self.episode_max_length = 150       # Maximum episode length

        self.num_serv = 3                   # Number of servers
        self.num_wq = 10                    # Number of works in the visible queue
        self.num_prio = 3                   # Number of different priorities in queue

        self.time_horizon = 20              # Number of timesteps in the graph
        self.max_job_len = 15               # Maximum duration of new job

        self.backlog_size = 60              # Backlog queue size
        self.max_track_since_new = 10       # Track how many timesteps since last new job

        self.new_job_rate = 0.8             # lambda in new job arrival Poisson Process
        self.lamda = 3                      # lambda for poisson distribution
        self.max_job_cnt = 5                # max number of jobs in one timestep

        self.unseen = True                  # seen or new examples

        self.work_dist = job_distribution.Dist(self.max_job_len)

        self.hold_penalty = -1         # penalty for holding things in the new work screen
        self.dismiss_penalty = -1      # penalty for missing a job because the queue is full
        self.delay_penalty = -1        # penalty for delaying things in the current work screen

        # Initial state (congestion) of the server params
        # ----------------------------
        # ----------------------------
        # probability of a server being crowded
        self.crowded_p = 0.3
        # maximum traffic of jobs in a crowded server
        self.max_crowded_congestion = 15
        # minimum traffic of jobs in a crowded server
        self.min_crowded_congestion = 10
        # maxmimum traffic of jobs in an uncrowded server
        self.max_uncrowded_congestion = 5
        # ----------------------------
        # ----------------------------

        # self.eval_seeds = [1, 26, 33, 59, 63, 32, 86, 93, 44, 77]

        self.vf_net = [256, 256, 256, 256]
        self.pi_net = [256, 256, 256, 256]
        self.policy_kwargs = {
            "net_arch": [{
                "vf": self.vf_net,
                "pi": self.pi_net
            }]
        }
