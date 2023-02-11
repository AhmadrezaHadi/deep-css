import os
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from .parameters import Parameters


class Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, pa, render=False,
                 repre='compact', end='all_done', seed=33) -> None:
        super(Env, self).__init__()

        self.pa = pa
        self.env_seed = seed
        if not self.pa.unseen:
            np.random.seed(self.env_seed)
        self.action_space = spaces.MultiDiscrete([self.pa.num_wq + 1,
                                                  self.pa.num_prio,
                                                  self.pa.num_serv])
        if repre == 'compact':
            self.observation_space = spaces.Box(low=0,
                                                high=1,
                                                shape=(self.pa.time_horizon * self.pa.num_serv +
                                                       self.pa.time_horizon * self.pa.num_serv +
                                                       self.pa.num_wq + 2,),
                                                dtype=np.float64)

        self.env_render = render
        self.repre = repre      # image or compact representation
        self.end = end          # termination type, 'no_new_job' or 'all_done'

        self.curr_time = 0
        self.work_dist = self.pa.work_dist.exp_model_dist

        # work sequence
        self.work_len_seqs = self.generate_work_sequence(
            simu_len=self.pa.simu_len)
        self.work_len_seqs.insert(0, [])
        self.env_len = len(self.work_len_seqs)

        self.seq_idx = 0
        # Initialize System
        self.machine = Machine(pa)
        self.job_record = self.machine.fill_up_servers()
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.extra_info = ExtraInfo(pa)

    def step(self, a):
        """
        TODO complete here
        """
        action = a[0]
        priority = a[1]
        server = a[2]
        status = None
        done = False
        reward = 0
        info = None
        if action == self.pa.num_wq:             # Explicit void action
            status = 'MoveOn'
        elif self.job_slot.slot[action] is None:      # Implicit void action
            status = 'MoveOn'
        else:
            self.job_slot.slot[action].priority = priority
            allocated = self.machine.allocate_job(self.job_slot.slot[action],
                                                  server)
            status = 'Allocate' if allocated else 'MoveOn'
        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            self.extra_info.time_proceed()

            # add new jobs
            self.seq_idx += 1
            if self.end == 'no_new_job':
                if self.seq_idx >= self.env_len:
                    done = True
            elif self.end == "all_done":  # everything has to be finished
                if self.seq_idx >= self.env_len and \
                   self.all_servers_empty() and \
                   all(s is None for s in self.job_slot.slot) and \
                   all(s is None for s in self.job_backlog.backlog):
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    done = True

            if not done:
                if self.seq_idx < self.env_len:
                    new_jobs = self.get_new_job_from_seq(self.seq_idx)

                    for job in new_jobs:     # a new job comes
                        to_backlog = True

                        for i in range(self.pa.num_wq):
                            if self.job_slot.slot[i] is None:
                                self.job_slot.slot[i] = job
                                # self.job_record.record[new_job.id] = new_job
                                to_backlog = False
                                break

                        if to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size:
                                self.job_backlog.backlog[self.job_backlog.curr_size] = job
                                self.job_backlog.curr_size += 1
                                # self.job_record.record[new_job.id] = new_job
                            else:   # abort, backlog is full
                                print("Backlog is full.")
                                del self.job_record.record[job.id]

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

        if self.env_render:
            self.plot_state()

        information = {}
        information['job_record'] = info
        return ob, reward, done, information

    def get_reward(self):
        """
        reward function for the environment
        """
        reward = 0
        for serv in self.machine.running_jobs:
            for k in serv:
                for job in serv[k]:
                    reward += self.pa.delay_penalty / float(job.len)

        for job in self.job_slot.slot:
            if job is not None:
                reward += self.pa.hold_penalty / float(job.len)

        for job in self.job_backlog.backlog:
            if job is not None:
                reward += self.pa.dismiss_penalty / float(job.len)

        return reward

    def observe(self):
        if self.repre == "compact":
            compact_repre = np.zeros((self.pa.time_horizon * self.pa.num_serv) +     # servers - works
                                     # servers - priorities
                                     (self.pa.time_horizon * self.pa.num_serv) +
                                     self.pa.num_wq +                                # work queue
                                     2,                                              # backlog and extra info indicator
                                     dtype=np.float64)
            running_jobs = self.machine.running_jobs
            job_slot = self.job_slot.slot
            backlog_curr_size = self.job_backlog.curr_size
            backlog_size = self.pa.backlog_size
            extra_info = self.extra_info.time_since_last_new_job

            work_queue = np.zeros((self.pa.num_wq, 1),
                                  dtype=np.float64)
            srv_works = np.zeros((self.pa.time_horizon, self.pa.num_serv),
                                 dtype=np.float64)
            srv_prios = np.zeros((self.pa.time_horizon, self.pa.num_serv),
                                 dtype=np.float64)

            for idx, serv in enumerate(running_jobs):
                for key in serv:
                    for ptr, job in enumerate(serv[key]):
                        srv_works[ptr, idx] += job.remaining_time / \
                            self.pa.max_job_len
                        srv_prios[ptr, idx] += (job.priority+1) / \
                            self.pa.num_prio

            for idx, job in enumerate(job_slot):
                if job is not None:
                    work_queue[idx] = job.len / self.pa.max_job_len
            ptr = 0
            srv_works = srv_works.flatten('F')
            srv_prios = srv_prios.flatten('F')
            work_queue = work_queue.flatten()
            compact_repre[ptr: srv_works.shape[0]] = srv_works
            ptr += srv_works.shape[0]
            compact_repre[ptr: ptr+srv_prios.shape[0]] = srv_prios
            ptr += srv_prios.shape[0]
            compact_repre[ptr: ptr+work_queue.shape[0]] = work_queue
            ptr += work_queue.shape[0]
            compact_repre[ptr] = backlog_curr_size / backlog_size
            compact_repre[ptr+1] = extra_info / \
                self.extra_info.max_tracking_time_since_last_job
            return compact_repre
        if self.repre == "image":
            pass

    def render(self):
        pass

    def all_servers_empty(self):
        """whether all servers are empty or not"""
        empty = True
        for dic in self.machine.running_jobs:
            for k in dic:
                if dic[k]:
                    empty = False
                    break
            if not empty:
                break
        return empty

    def generate_work_sequence(self, simu_len):
        """
        generates a sequence of works
        """
        work_len_seq = []
        size = 0
        while size < simu_len:
            cnt = np.random.poisson(self.pa.lamda)
            if cnt + size > simu_len:
                continue
            size += cnt
            work_len_seq.append([self.work_dist() for _ in range(cnt)])

        return work_len_seq

    def get_new_job_from_seq(self, seq_index):
        jobs = []
        for l in self.work_len_seqs[seq_index]:
            new_job = Job(job_id=len(self.job_record.record),
                          job_len=l,
                          enter_time=self.curr_time)
            self.job_record.record[new_job.id] = new_job
            jobs.append(new_job)
        return jobs

    def plot_state(self):
        os.system('clear')
        for idx, serv in enumerate(self.machine.running_jobs):
            print(f"server no.: {idx}")
            for k in serv:
                print(f"\tpriority: {k}")
                for job in serv[k]:
                    if job is not None:
                        print(f"\t\tJob ID: ", job.id, end='')
                        print(f"\tRemaining Time: ", job.remaining_time)
        for job in self.job_slot.slot:
            job: Job
            if job is not None:
                print("Job ID: ", job.id, end='')
                print(f"\tJob Length: ", job.remaining_time)
            else:
                print("None")

        print("current backlog size: ", self.job_backlog.curr_size)
        print("time since last new job: ",
              self.extra_info.time_since_last_new_job)
        # TODO complete here

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        if not self.pa.unseen:
            np.random.seed(self.env_seed)

        self.work_len_seqs = self.generate_work_sequence(
            simu_len=self.pa.simu_len)
        self.work_len_seqs.insert(0, [])
        self.env_len = len(self.work_len_seqs)

        # initialize system
        self.machine = Machine(self.pa)
        self.job_record = self.machine.fill_up_servers()
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
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

    def __init__(self, job_len, job_id, enter_time, priority=-1) -> None:
        self.id = job_id
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1
        self.finish_time = -1
        self.priority = priority
        self.remaining_time = job_len

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
    def __init__(self, pa: Parameters) -> None:
        self.num_serv = pa.num_serv
        self.time_horizon = pa.time_horizon
        self.crowded_p = pa.crowded_p
        self.min_crowded = pa.min_crowded_congestion
        self.max_crowded = pa.max_crowded_congestion
        self.max_uncrowded = pa.max_uncrowded_congestion
        self.num_prio = pa.num_prio

        # free slots in each server
        self.avlbl_slots = np.array(
            [self.time_horizon for _ in range(self.num_serv)])
        # running jobs in each server
        self.running_jobs = [{prio: []
                              for prio in range(pa.num_prio)} for _ in range(self.num_serv)]

        # TODO Graphical Repre

    def fill_up_servers(self):
        """
        Fills the servers with jobs having random priorities and lengths for initial state of the environment.
        returns servers state dictionary.
        """
        job_record = JobRecord()
        crowded = False
        for idx in range(self.num_serv):
            if np.random.random() < self.crowded_p:
                # crowded
                crowded = True
                congestion = np.random.randint(
                    self.min_crowded, self.max_crowded + 1)
            else:
                # uncrowded
                congestion = np.random.randint(0, self.max_uncrowded + 1)

            avlbl_slots = self.time_horizon - congestion

            while self.avlbl_slots[idx] > avlbl_slots:
                work_len = \
                    np.random.randint(1, self.max_crowded) if crowded \
                    else np.random.randint(1, self.max_uncrowded)
                if self.avlbl_slots[idx] - work_len < avlbl_slots:
                    continue
                prio = np.random.randint(1, self.num_prio)
                new_job = Job(job_id=len(job_record.record),
                              job_len=work_len,
                              enter_time=0,
                              priority=prio)  # initial state of the environment, thus enter_time = 0
                job_record.record[new_job.id] = new_job
                self.avlbl_slots[idx] -= work_len
                self.running_jobs[idx][prio].append(new_job)
        return job_record

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
