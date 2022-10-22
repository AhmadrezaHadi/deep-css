from typing import Callable
from envs.deepcss_v0.environment import Job, Env
from envs.deepcss_v0.parameters import Parameters
from stable_baselines3.common.monitor import Monitor


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


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
        current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > 0.25:
            return progress_remaining * initial_value
        else:
            return final_value

    return func


def eval_model(model, env, iters):
    methods = ['Random', 'Model Algorithm']
    random_mean = 0
    ml_mean = 0
    ITERS = iters
    for _ in range(ITERS):
        for m in methods:
            obs = env.reset()
            while True:
                if m == 'Random':
                    action = env.action_space.sample()
                else:
                    action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action)
                if done:
                    if m == 'Random':
                        random_mean += calculate_average_slowdown(info)
                    else:
                        ml_mean += calculate_average_slowdown(info)
                    break
    print(f"Random: {random_mean/ITERS :.2f}")
    print(f"Model: {ml_mean/ITERS : .2f}")


def make_env(seed):
    pa = Parameters()
    pa.unseen = False

    def _init():
        env = Monitor(Env(pa, seed=seed))
        return env

    return _init
