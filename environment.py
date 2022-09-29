from hashlib import new
from os import stat
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
                 repre='compact', end='all_done', seen=False) -> None:
        super(Env, self).__init__()

        self.pa = pa
        if not self.pa.unseen:
            np.random.seed(33)
        self.action_space = spaces.Box(low=np.array([0, 0, 0]),
                                       high=np.array([self.pa.num_wq,
                                                      self.pa.num_prio, self.pa.num_serv]),
                                       dtype=np.uint8)
        # if repre == 'compact':
        # TODO complete here
        # self.observation_space = spaces.Box()

        self.render = render
        self.repre = repre      # image or compact representation
        self.end = end          # termination type, 'no_new_job' or 'all_done'

        self.curr_time = 0
        self.work_dist = self.pa.work_dist.bi_model_dist

        # work sequence
        self.work_len_seqs = self.generate_work_sequence(
            simu_len=self.pa.simu_len)

        self.seq_idx = 0
        # Initialize System
        self.machine = Machine(pa)
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)

    def step(self, a):
        action = a[0]
        priority = a[1]
        server = a[2]
        status = None
        done = False
        reward = 0
        info = None
        print(type(action))
        print('here')
        if action == self.pa.num_wq:             # Explicit void action
            status = 'MoveOn'
        elif self.job_slot.slot[action] is None:      # Implicit void action
            status = 'MoveOn'
        else:
            self.job_slot.slot[action].priority = priority
            allocated = self.machine.allocate_job(self.job_slot[action],
                                                  server)
            status = 'Allocate' if allocated else 'MoveOn'
        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            self.extra_info.time_proceed()

            # add new jobs
            self.seq_idx += 1
            if self.end == 'no_new_job':
                if self.seq_idx >= self.pa.simu_len:
                    done = True
            elif self.end == "all_done":  # everything has to be finished
                if self.seq_idx >= self.pa.simu_len and \
                   len(self.machine.running_job) == 0 and \
                   all(s is None for s in self.job_slot.slot) and \
                   all(s is None for s in self.job_backlog.backlog):
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    done = True

            if not done:

                if self.seq_idx < self.pa.simu_len:
                    new_job = self.get_new_job_from_seq(self.seq_idx)

                if new_job.len > 0:     # a new job comes
                    to_backlog = True

                    for i in range(self.pa.num_wq):
                        if self.job_slot.slot[i] is None:
                            self.job_slot.slot[i] = new_job
                            self.job_record.record[new_job.id] = new_job
                            to_backlog = False
                            break

                    if to_backlog:
                        if self.job_backlog.curr_size < self.pa.backlog_size:
                            self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                            self.job_backlog.curr_size += 1
                            self.job_record.record[new_job.id] = new_job
                        else:   # abort, backlog is full
                            print("Backlog is full.")

                    self.extra_info.new_job_comes()
            reward = self.get_reward()
        elif status == 'Allocate':
            self.job_record.record[self.job_slot.slot[action].id] = self.job_slot.slot[action]
            self.job_slot.slot[action] = None

            # dequeue backlog
            if self.job_backlog.curr_size > 0:
                self.job_slot.slot[action] = self.job_backlog.backlog[0]
                self.job_backlog.backlog[:-1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1

        ob = self.observe()
        info = self.job_record

        if self.render:
            self.plot_state()

        information = {}
        information['job_record'] = info
        return ob, reward, done, information

    def get_reward(self):
        pass

    def observe(self):
        pass

    def render(self):
        pass

    def generate_work_sequence(self, simu_len):
        """
        generates a sequence of works
        """
        work_len_seq = np.zeros(simu_len, dtype=np.int8)

        for i in range(simu_len):
            if np.random.rand() < self.pa.new_job_rate:
                work_len_seq[i] = self.work_dist()

        return work_len_seq

    def get_new_job_from_seq(self, seq_index):
        new_job = Job(job_id=len(self.job_record.record),
                      job_len=self.work_len_seqs[seq_index],
                      enter_time=self.curr_time)
        return new_job

    def plot_state(self):
        pass

    def reset(self):
        # return super().reset()
        self.seq_idx = 0
        self.curr_time = 0

        if not self.pa.unseen:
            np.random.seed(33)

        self.work_len_seqs = self.generate_work_sequence(
            simu_len=self.pa.simu_len)

        # initialize system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)

        return self.observe()

    def close(self) -> None:
        return super().close()


class Job:
    """
    Structure of a single job in the queue.
    id : id of the job
    len : length of the job
    enter_time : time that job entered the queue
    start_time: time that machine started to do the job
    finish_time: timestep that job finished
    priority : priority of the job (0 is the highest).
    remaining_time : remaining time till job finishes.
    """

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
    """
    list of jobs in the working queue
    """

    def __init__(self, pa) -> None:
        self.slot = [None] * pa.num_wq


class JobBacklog:
    """
    list of jobs in the backlog
    """

    def __init__(self, pa) -> None:
        self.backlog = [None] * pa.backlog_size
        self.curr_size = int(0)


class JobRecord:
    """
    record of all jobs in simulation
    """

    def __init__(self) -> None:
        self.record = {}


class Machine:
    def __init__(self, pa) -> None:
        self.num_serv = pa.num_serv
        self.time_horizon = pa.time_horizon

        # free slots in each server
        self.avlbl_slots = np.array(
            [self.time_horizon for _ in range(self.num_serv)])
        # running jobs in each server
        self.running_jobs = [{prio: []
                              for prio in range(pa.num_prio)} for _ in range(self.num_serv)]

        # TODO Graphical Repre

    def allocate_job(self, job: Job, num_serv):
        """
        Tries to allocate job in the running_jobs list from the num_serv
        if it's successfull, returns True, False otherwise.
        """
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
        """
        proceeds one timestep forward in the machine.
        set start time for the jobs in queue(if it's not set yet)
        moves one step forward.
        """
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
