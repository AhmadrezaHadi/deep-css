import argparse
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from environment import Env
from parameters import Parameters
from utils import calculate_average_slowdown, SJF

#####################################################################################
##########################            Arguments            ##########################
#####################################################################################
parser = argparse.ArgumentParser(description="")
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
parser.add_argument("-l", "--load", type=str, help="model path to load")

args = parser.parse_args()

RENDER = args.render
TIMESTEPS = args.timesteps
REPRE = args.representation


def eval_model(model, env):
    methods = ['Random', 'SJF', 'Model Algorithm']
    for m in methods:
        obs = env.reset()
        while True:
            if m == 'Random':
                action = env.action_space.sample()
            elif m == 'SJF':
                # for _ in range(5):
                # obs = env.step(5)
                action = SJF(env)
                # break
            else:
                action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            if done:
                print('done')
                print(
                    f"{m} method average slowdown = {calculate_average_slowdown(info=info)}")
                break


if __name__ == '__main__':
    pa = Parameters()
    env = Env(pa)
    env.reset()

    eval_pa = Parameters()
    eval_pa.unseen = False
    eval_env = Env(pa)

    net = [256, 256, 256, 256, 256, 256, 256, 256]
    policy_kwargs = {
        "net_arch": [{
            "vf": net,
            "pi": net
        }]
    }

    if args.mode == 'train':
        checkpoint_callback = CheckpointCallback(save_freq=20000,
                                                 save_path=f'./models/{args.algorithm}_256neurons_8layer_1',
                                                 name_prefix=f'{args.algorithm}')
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                     log_path='./logs/', eval_freq=500, deterministic=True, render=False)
        callbacks = CallbackList([checkpoint_callback, eval_callback])

        if args.algorithm == 'ppo':
            print('creating model')
            model = PPO('MlpPolicy', env,
                        tensorboard_log='./tensorboard/', device='auto', policy_kwargs=policy_kwargs)
            # model = model.load(
            #     'models/ppo_128neurons_3layer/ppo_2300000_steps', env)
            try:
                print("training")
                model.learn(TIMESTEPS, callback=callbacks,
                            tb_log_name=f'{args.algorithm}_distinct_policy_net_256_8layer')
            except:
                # model.save_replay_buffer('tmp/last_model.pkl')
                model.save('tmp/last_model')
                print(f"model trained using {args.algorithm} algorithm")
        elif args.algorithm == 'dqn':
            pass

    if args.mode == 'eval':
        if args.algorithm == 'ppo':
            if not args.load:
                raise "model path not specified (--load model_path)"
            eval_env.reset()
            model = PPO('MlpPolicy', env).load(args.load, eval_env)
            eval_model(model, env)
        elif args.algorithm == 'dqn':
            pass
