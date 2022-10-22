from re import sub
from typing import Callable
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from utils import calculate_average_slowdown, SJF, make_env
from envs.deepcss_v0.environment import Env
from envs.deepcss_v0.environment import Parameters
import gym
import envs

#####################################################################################
##########################            Arguments            ##########################
#####################################################################################
parser = argparse.ArgumentParser(description="")
# parser.add_argument("module", choices=['rllib', 'sb3'],
#                     help="module for implementing the algorithm.")
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

if __name__ == '__main__':
    pa = Parameters()

    # envs = make_vec_env('deepcss-v0', n_envs=CPU, vec_env_cls=SubprocVecEnv)
    env = gym.make('deepcss-v0')

    eval_envs = []
    for seed in pa.eval_seeds:
        eval_envs.append(make_env(seed=seed))
    eval_envs = DummyVecEnv(eval_envs)

    # eval_kwargs = {"pa": eval_pa}
    # eval_env = make_vec_env('deepcss-v0', 1, seed=33, env_kwargs=eval_kwargs)

    policy_kwargs = pa.policy_kwargs

    if args.mode == 'train':
        if not args.name:
            print("Please set a name for training model")
            exit(0)
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
            model = PPO('MlpPolicy', env,
                        tensorboard_log='./tensorboard/', device='cuda:0',
                        policy_kwargs=policy_kwargs)
            if args.load:
                print(f"loading model from: {args.load}")
                model = model.load(args.load, env)
            try:
                print("training")
                model.learn(TIMESTEPS, callback=callbacks,
                            tb_log_name=f'{args.algorithm}_{MODEL_NAME}')
            except:
                model.save('tmp/last_model')
                print(f"model trained using {args.algorithm} algorithm.")
        elif args.algorithm == 'dqn':
            pass

    if args.mode == 'eval':
        if args.algorithm == 'ppo':
            if not args.load:
                print("model path not specified (--load model_path)")
                exit(0)
            # eval_env.reset()
            # model = PPO('MlpPolicy', eval_env).load(args.load, eval_env)
            # eval_model(model, eval_env, args.iters)
        elif args.algorithm == 'dqn':
            pass
