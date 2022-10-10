import numpy as np
import job_distribution


class Parameters:
    def __init__(self) -> None:

        self.simu_len = 30              # Simulation length
        self.episode_max_length = 450   # Maximum episode length

        self.num_serv = 3               # Number of servers
        self.num_wq = 10                # Number of works in the visible queue
        self.num_prio = 4               # Number of different priorities in queue

        self.time_horizon = 20          # Number of timesteps in the graph
        self.max_job_len = 15           # Maximum duration of new job

        self.backlog_size = 60          # Backlog queue size
        self.max_track_since_new = 10   # Track how many timesteps since last new job

        self.new_job_rate = 0.7         # lambda in new job arrival Poisson Process
        self.new_job_cnt_mean = 2       # mean for new job count
        self.new_job_cnt_std = 1        # standard deviation for new jobs count
        self.max_job_cnt = 5            # max num of jobs in one timestep

        self.unseen = True              # seen or new examples

        self.work_dist = job_distribution.Dist(self.max_job_len)

        self.hold_penalty = -1         # penalty for holding things in the new work screen
        self.dismiss_penalty = -1      # penalty for missing a job because the queue is full
        self.delay_penalty = -1        # penalty for delaying things in the current work screen
