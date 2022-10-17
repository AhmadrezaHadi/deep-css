from ast import arg
from typing import Callable
import argparse
from cmath import inf
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from environment import Env
from parameters import Parameters
from utils import calculate_average_slowdown, SJF
import gym

#####################################################################################
##########################            Arguments            ##########################
#####################################################################################
parser = argparse.ArgumentParser(description="")
parser.add_argument("module", choices=['rllib', 'sb3'],
                    help="module for implementing the algorithm.")
parser.add_argument("mode", choices=['train', 'eval'],
                    help="Train or Eval model")
parser.add_argument("algorithm", choices=['ppo', 'dqn'],
                    help="algorithm for training")
parser.add_argument("-t", "--timesteps", default=10_000_000, type=int,
                    help="Timesteps for training")
parser.add_argument("-r", "--render", default=False, type=bool,
                    help="Render the output of environment or not")
parser.add_argument("-re", "--representation", default='compact', type=str,
                    help='state returned from the environment (compact or image).')
parser.add_argument("-l", "--load", type=str,
                    help="model path to load")
parser.add_argument("-i", "--iters", type=int, default=1,
                    help="iterations for evaluation")
parser.add_argument("-n", "--name", type=str,
                    help="name for the training model")
parser.add_argument("-u", "--unseen", type=bool, default=False,
                    help="Whether to set a fixed or random seed for evaluation.")
parser.add_argument("-c", "--cpu", default=4, type=int,
                    help="Number of cpus, for multiprocessing the learnign process")

args = parser.parse_args()

RENDER = args.render
TIMESTEPS = args.timesteps
REPRE = args.representation
UNSEEN = args.unseen
CPU = args.cpu


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


def make_env(pa, seed=None) -> Callable:
    def _init() -> gym.Env:
        if seed is not None:
            env = Monitor(Env(pa, seed=seed))
        else:
            env = Env(pa)
        return env
    return _init


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


if __name__ == '__main__':
    pa = Parameters()
    # env = Env(pa)
    # env.reset()
    envs = SubprocVecEnv([make_env(pa) for _ in range(CPU)])

    eval_pa = Parameters()
    eval_pa.unseen = UNSEEN
    # eval_env = Monitor(Env(eval_pa))
    eval_envs = []
    for seed in [1, 26, 33, 59, 63, 32, 86, 93, 44, 77]:
        eval_envs.append(make_env(eval_pa, seed=seed))
    eval_envs = SubprocVecEnv(eval_envs)

    policy_kwargs = pa.policy_kwargs

    if args.module == 'sb3':
        if args.mode == 'train':
            if not args.name:
                raise "Please set a name for training model"
            else:
                MODEL_NAME = args.name
            checkpoint_callback = CheckpointCallback(save_freq=5_000,
                                                     save_path=f'./models/{args.algorithm}_{MODEL_NAME}',
                                                     name_prefix=f'{args.algorithm}')
            eval_callback = EvalCallback(eval_envs, best_model_save_path='./logs/',
                                         log_path='./logs/', eval_freq=2_500, deterministic=True, render=False)
            callbacks = CallbackList([checkpoint_callback, eval_callback])

            if args.algorithm == 'ppo':
                print('creating model')
                model = PPO('MlpPolicy', envs,
                            tensorboard_log='./tensorboard/', device='auto',
                            policy_kwargs=policy_kwargs)
                if args.load:
                    print(f"loading model from: {args.load}")
                    model = model.load(args.load, envs)
                try:
                    print("training")
                    model.learn(TIMESTEPS, callback=callbacks,
                                tb_log_name=f'{args.algorithm}_{MODEL_NAME}')
                except:
                    model.save('tmp/last_model')
                    print(f"model trained using {args.algorithm} algorithm.")
            elif args.algorithm == 'dqn':
                pass
    elif args.module == 'rllib':
        pass

    if args.mode == 'eval':
        if args.algorithm == 'ppo':
            if not args.load:
                raise "model path not specified (--load model_path)"
            eval_env.reset()
            model = PPO('MlpPolicy', eval_env).load(args.load, eval_env)
            eval_model(model, eval_env, args.iters)
        elif args.algorithm == 'dqn':
            pass
