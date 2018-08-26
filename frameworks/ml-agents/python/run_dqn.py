# # Unity ML Agents
# ## ML-Agent Learning

from baselines import deepq
from baselines.common.atari_wrappers import WarpFrame, FrameStack

from unityagents import UnityEnvironment, UnityEnvironmentException

from baselines_wrapper import FloatToUInt8Frame, MLToGymEnv

import tensorflow as tf
import numpy as np
import time
import keyboard
import cv2

def _make_dqn(unity_env, train_mode, reward_range=(-np.inf, np.inf)):
    env = MLToGymEnv(unity_env, train_mode=train_mode, reward_range=reward_range)
    env = FloatToUInt8Frame(env)
    env = WarpFrame(env) # Makes sure we have 84 x 84 b&w
    env = FrameStack(env, 4) # Stack last 4 frames
    return env

def _create_summary_callback(summary_writer):
    def _summary_callback(local_vars, global_vars):
        if (local_vars['done']):
            t = local_vars['t']
            episode_rewards = local_vars['episode_rewards']
            summary = tf.Summary()
            summary.value.add(tag='Info/Mean 100 Reward', simple_value=round(np.mean(episode_rewards[-101:-1]), 1))
            summary.value.add(tag='Info/Episode Length', simple_value=local_vars['episode_length'])
            summary.value.add(tag='Info/Episode Reward', simple_value=episode_rewards[-2])
            summary.value.add(tag='Info/Episode Count', simple_value=(len(episode_rewards) - 1))
            summary.value.add(tag='Info/% time spent Exploring', simple_value=int(100 * local_vars['exploration'].value(t)))

            summary_writer.add_summary(summary, t)
            summary_writer.flush()
        return False # we only want to log here, no need to stop algorithm

    return _summary_callback

def learn(env_path, seed, max_steps, reward_range, base_port, unity_arguments, summary_writer):
    unity_env = UnityEnvironment(file_name=env_path, seed=seed, base_port=base_port, arguments=unity_arguments)
    env = _make_dqn(unity_env, train_mode=True, reward_range=reward_range)   

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
    )

    act = deepq.learn(
        env,
        q_func=model,
        lr=0.0000625,
        max_timesteps=max_steps,
        buffer_size=200000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=80000,
        target_network_update_freq=32000,
        gamma=0.99,
        print_freq=1,
        prioritized_replay=True,
        prioritized_replay_beta0=0.4,
        param_noise=False,
        double_q=True,
        callback=_create_summary_callback(summary_writer),
    )

    env.close()
    return act

def enjoy(env_path, seed, max_steps, base_port, unity_arguments, model_path):
    unity_env = UnityEnvironment(file_name=env_path, seed=seed, base_port=base_port, arguments=unity_arguments)
    env = _make_dqn(unity_env, train_mode=False)
    act = deepq.load(model_path)

    step_count = 0

    while step_count < max_steps:
        obs, done = env.reset(), False
        episode_rew = 0
        time.sleep(2) # Time delay after reset for player interaction

        while not done:
            if keyboard.is_pressed('n'):
                break

            step_count += 1
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew

        print("Episode reward: ", episode_rew)
        print("Elapsed steps: ", step_count)

    env.close()
