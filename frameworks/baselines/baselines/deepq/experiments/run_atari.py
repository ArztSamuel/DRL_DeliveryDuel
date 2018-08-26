from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari

from pynput import keyboard

def callback(lcl, _glb):
    global abort_training

    # stop training on input
    return abort_training

def on_press(key):
    global abort_training
    global q_pressed

    if key == keyboard.KeyCode.from_char('q'):
        q_pressed = True
    elif key == keyboard.Key.esc and q_pressed:
        print("Aborting Training...")
        abort_training = True
        return False

def on_release(key):
    global q_pressed

    if key == keyboard.KeyCode.from_char('q'):
        q_pressed = False

def main():
    global abort_training
    global q_pressed
    
    abort_training = False
    q_pressed = False
    
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        print_freq=1,
        prioritized_replay=bool(args.prioritized),
        callback=callback
    )    
    print("Saving model to pong_model.pkl")
    act.save("pong_model.pkl")
    env.close()

if __name__ == '__main__':
    main()
            

