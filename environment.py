from queue import PriorityQueue
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


class Job:
    def __init__(self, job_len, job_id, enter_time) -> None:
        self.id = job_id
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1
        self.finish_time = -1
        self.priority = -1
        self.remaining_time = -1

    def __str__(self) -> str:
        return f"id={self.id}, len={self.len}, enter_time={self.enter_time}, prio={self.priority}, remain={self.remaining_time}"


class JobSlot:
    def __init__(self, pa) -> None:
        self.slot = [None] * pa.num_wq


class JobBacklog:
    def __init__(self, pa) -> None:
        self.backlog = [None] * pa.backlog_size
        self.curr_size = int(0)


class JobRecord:
    def __init__(self) -> None:
        self.record = {}


class Machine:
    def __init__(self, pa) -> None:
        self.num_serv = pa.num_serv
        self.time_horizon = pa.time_horizon

        self.avlbl_slots = np.array(
            [self.time_horizon for _ in range(self.num_serv)])
        self.running_jobs = [{prio: []
                              for prio in range(pa.num_prio)} for _ in range(3)]

        # TODO Graphical Repre

    def allocate_job(self, job: Job, num_serv):
        allocate = False
        prio = job.priority
        if job.len <= self.avlbl_slots[num_serv]:
            allocate = True
            self.avlbl_slots[num_serv] -= job.len
            if self.running_jobs[num_serv][prio] is None:
                self.running_jobs[num_serv][prio] = [job, ]
            else:
                self.running_jobs[num_serv][prio].append(job)

            # TODO update graphical repre

        return allocate

    def time_proceed(self, curr_time):
        prev_time = curr_time - 1
        for idx in range(len(self.running_jobs)):
            serv = self.running_jobs[idx]
            for k in sorted(serv):
                if serv[k]:
                    job = serv[k].pop(0)
                    # check if the job is scheduled yet
                    if job.start_time == -1:
                        job.start_time = prev_time
                        job.remaining_time = job.len
                    # decrease the remaining time
                    if job.remaining_time > 0:
                        job.remaining_time -= 1
                        self.avlbl_slots[idx] += 1
                    # check if the job has finished
                    if job.remaining_time == 0:
                        job.finish_time = curr_time
                    else:
                        serv[k].insert(0, job)
                    break

        # TODO update graphical repre


class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1
