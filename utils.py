from cmath import inf
from environment import Job
from environment import Env


def calculate_average_slowdown(info):
    job_rec = info['job_record'].record
    average_slowdown = 0
    len = 0
    for index in job_rec:
        len += 1
        job = job_rec[index]
        average_slowdown += (job.finish_time - job.enter_time) / job.len
    average_slowdown /= len
    return average_slowdown


def SJF(env: Env):
    action = None
    len = 200
    for idx, job in enumerate(env.job_slot.slot):
        if job is not None:
            if job.len < len:
                action = idx
    if action is None:
        return 0
    return action
