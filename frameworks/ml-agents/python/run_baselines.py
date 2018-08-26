
import logging

import os
from docopt import docopt

from baselines.common import set_global_seeds

import numpy as np
import tensorflow as tf

from time import strftime
from run_dqn import learn as learn_dqn, enjoy as enjoy_dqn
from run_a2c import learn as learn_a2c, enjoy as enjoy_a2c

if __name__ == '__main__':
    logger = logging.getLogger("unityagents")
    _USAGE = '''
    Usage:
      learn_baselines (<env>) [options]
      learn_baselines --help

    Options:
      --method=<n>                  The method to be used for training [choices: dqn, a2c, acktr][default: dqn].
      --seed=<n>                    Random seed used for training [default: -1].
      --enjoy=<n>                   The path of the model to load in 'enjoy' mode, i.e. only serves as an actor without training [default: None].
      --rewardLowerBounds=<n>       The lower bounds of the rewards of the environment [default: -inf].
      --rewardUpperBounds=<n>       The upper bounds of the rewards of the environment [default: inf].
      --max-steps=<n>               The amount of timesteps before the learning process is stopped [default: 10000000].
      --base-port=<n>               The base port to be used for communication between python and the environment [default: 5005].
      --output-folder=<n>           The folder to save the trained model and summary data to [default: outputs\\].
      --unity-arguments=<n>         The arguments to pass to the started unity process [default: '']
    '''

    options = docopt(_USAGE)
    logger.info(options)

    # General parameters
    method = options['--method']
    env_path = options['<env>']
    seed = int(options['--seed'])
    enjoy_path = options['--enjoy']
    reward_range = (-np.inf, np.inf)
    if options['--rewardLowerBounds'] != '-inf':
        reward_range = (float(options['--rewardLowerBounds']), reward_range[1])
    if options['--rewardUpperBounds'] != 'inf':
        reward_range = (reward_range[0], float(options['--rewardUpperBounds']))
    if seed == -1:
        seed = np.random.randint(0, 999999)
    max_steps = int(options['--max-steps'])
    base_port = int(options['--base-port'])
    output_folder = options['--output-folder']
    time_string = strftime("%Y-%m-%d.%H-%M-%S")
    unity_arguments = options['--unity-arguments']
    if unity_arguments == '':
        unity_arguments = None
    else:
        unity_arguments = unity_arguments.split(" ")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    summary_writer = tf.summary.FileWriter(output_folder + "summaries\\" + method + "\\" + time_string + "\\")

    set_global_seeds(seed)

    if enjoy_path == 'None':
        # Train a new model
        act = None
        if method == 'dqn':
            print("Training using DQN...")
            act = learn_dqn(env_path=env_path, seed=seed, max_steps=max_steps, reward_range=reward_range, base_port=base_port, unity_arguments=unity_arguments, summary_writer=summary_writer)
        elif method == 'a2c':
            print("Training using A2C...")
            act = learn_a2c(env_path=env_path, seed=seed, max_steps=max_steps, reward_range=reward_range, base_port=base_port, unity_arguments=unity_arguments, summary_writer=summary_writer)
        else:
            print("Unknown method: \"" + method + "\".")

        model_file_name = os.path.basename(env_path).split('.')[0] + ".pkl"
        model_path = os.path.abspath(os.path.dirname(__file__)) + "\\" + output_folder + "models\\" + method + "\\" + time_string + "\\"
        print("Saving model to " + model_path + ".")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        act.save(model_path + model_file_name)
    else:
        # Load and enjoy an existing model
        if method == 'dqn':
            print("Enjoying using DQN...")
            enjoy_dqn(env_path=env_path, seed=seed, max_steps=max_steps, base_port=base_port, unity_arguments=unity_arguments, model_path=enjoy_path)
        elif method == 'a2c':
            print("Loading A2C models not supported yet...")
        else:
            print("Unknown method: \"" + method + "\".")