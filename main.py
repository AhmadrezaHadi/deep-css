import gym
import envs
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from utils import calculate_average_slowdown, SJF, make_env
from arg_loader import arg_parser
from envs.deepcss_v0.environment import Parameters as V0_PA
from envs.deepcss_v1.environment import Parameters as V1_PA
from envs.deepcss_v2.environment import Parameters as V2_PA


def make_eval_envs(pa):
    eval_envs = []
    for seed in pa.eval_seeds:
        eval_envs.append(make_env(seed=seed))
    eval_envs = DummyVecEnv(eval_envs)
    return eval_envs


def main():
    args = arg_parser()
    ENV_ID = args.environment
    TIMESTEPS = args.timesteps
    CPU = args.cpu
    CLIP_FN = get_schedule_fn(args.cliprange)

    vec_envs = make_vec_env(ENV_ID, n_envs=CPU)
    if ENV_ID == 'deepcss-v0':
        policy_kwargs = V0_PA().policy_kwargs
    elif ENV_ID == 'deepcss-v1':
        policy_kwargs = V1_PA().policy_kwargs
    elif ENV_ID == 'deepcss-v2':
        policy_kwargs = V2_PA().policy_kwargs
    else:
        print('Environment ID not found.')
        exit(0)

    if args.mode == 'train':
        if not args.name:
            print("Please set a name for training model")
            exit(0)
        else:
            MODEL_NAME = args.name
        checkpoint_callback = CheckpointCallback(save_freq=5_000,
                                                 save_path=f'./models/{MODEL_NAME}_{args.algorithm}',
                                                 name_prefix=f'{args.algorithm}')
        eval_callback = EvalCallback(vec_envs, best_model_save_path=f'./logs/{MODEL_NAME}/',
                                     log_path=f'./logs/{MODEL_NAME}/', eval_freq=2_500, deterministic=True, render=False)
        callbacks = CallbackList([checkpoint_callback, eval_callback])

        if args.algorithm == 'ppo':
            print('creating model')
            model = PPO('MlpPolicy', vec_envs, batch_size=args.batchsize,
                        tensorboard_log='./tensorboard/', device='auto',
                        clip_range=CLIP_FN, policy_kwargs=policy_kwargs)
            if args.load:
                print(f"loading model from: {args.load}")
                model = model.load(args.load, vec_envs)
                model.clip_range = CLIP_FN
            try:
                print(f"training on {model.device}")
                model.learn(TIMESTEPS, callback=callbacks,
                            tb_log_name=f'{MODEL_NAME}_{args.algorithm}')
            except:
                model.save(f'tmp/{MODEL_NAME}')
                print(f"model trained using {args.algorithm} algorithm.")
                exit(0)
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


if __name__ == '__main__':
    main()
