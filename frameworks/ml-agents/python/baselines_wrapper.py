import gym
import numpy as np
import cv2
from gym import spaces

class MLToGymEnv(gym.Env):
    def __init__(self, env, train_mode, reward_range=(-np.inf, np.inf)):
        """Wraps UnityEnvironment of ML-Agents to be used by baselines algorithms
        """
        gym.Env.__init__(self)

        self.unityEnv = env
        self.train_mode = train_mode
        self.reward_range = reward_range

        assert self.unityEnv.number_external_brains > 0, "No external brains defined in unityEnv"
        self.__externalBrainName = self.unityEnv.external_brain_names[0]
        externalBrain = self.unityEnv.brains[self.__externalBrainName]
        actionSpaceSize = externalBrain.vector_action_space_size
        assert actionSpaceSize > 0
        self.action_space = spaces.Discrete(actionSpaceSize)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8) # TODO actually read dimensions from brain info

        # TODO set observation space according to brain


    def step(self, action):
        action_vector = {}
        action_vector[self.__externalBrainName] = [action] # needs to be list in case of multiple agents, TODO: support more than one agent
        brain_infos = self.unityEnv.step(action_vector)
        brain_info = brain_infos[self.__externalBrainName]
        obs = brain_info.visual_observations[0][0]
        reward = brain_info.rewards[0]
        done = brain_info.local_done[0]
        info = None
        return obs, reward, done, info

    def reset(self):
        obs_dict = self.unityEnv.reset(train_mode=self.train_mode)
        # observations of used external brain -> visual observation -> of camera 0 of agent 0
        return obs_dict[self.__externalBrainName].visual_observations[0][0] 

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        return self.unityEnv.close()

    def seed(self, seed=None):
        raise NotImplementedError

class FloatToUInt8Frame(gym.ObservationWrapper):
    def __init__(self, env):
        """Convert observation image from float64 to uint8"""
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, frame):
        # convert from float64, range 0 - 1 to uint8, range 0 - 255
        frame = 255 * frame
        frame = frame.astype(np.uint8)
        frame = frame[...,::-1] #convert to bgr for opencv imshow
        return frame