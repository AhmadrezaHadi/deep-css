from argparse import ArgumentParser as AP


def arg_parser():
    parser = AP(description="")

    parser.add_argument("mode", choices=['train', 'eval'],
                        help="Train or Eval model")
    parser.add_argument("algorithm", choices=['ppo', 'dqn'],
                        help="algorithm for training")
    parser.add_argument("-t", "--timesteps", default=50_000_000, type=int,
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
    parser.add_argument("-c", "--cpu", default=2, type=int,
                        help="Number of cpus, for multiprocessing the learnign process")
    parser.add_argument("-cr", "--cliprange", default=0.2, type=float,
                        help="Clip range parameter for ppo model")
    parser.add_argument("-np", "--numprio", default=3, type=int,
                        help="Number of priorities in each server.")
    parser.add_argument("-ns", "--numserv", default=3, type=int,
                        help="Number of servers in the environment.")
    parser.add_argument("-bs", "--batchsize", default=64,
                        type=int, help="Mini-batch size for learning process.")
    parser.add_argument("-e", "--environment", default='deepcss-v0',
                        type=str, help="Environment ID for the training process.")

    return parser.parse_args()
