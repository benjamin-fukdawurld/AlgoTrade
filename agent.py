# Import Stable baselines stuffs (RL algorithms and tools)
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


from stable_baselines3.common.env_util import make_vec_env

from custom_env import CustomEnv

import numpy as np


class Agent:
    def __init__(self, data, frame_bound, window_size):
        self.data = data
        self._env_maker = lambda: CustomEnv(
            df=self.data, window_size=window_size, frame_bound=frame_bound)
        self._training_env = make_vec_env(self._env_maker, n_envs=4)

    @property
    def env_maker(self):
        return self._env_maker

    @property
    def training_env(self):
        return self._training_env

    def train(self):
        self._model = A2C('MlpPolicy', self.training_env, verbose=1)
        self._model.learn(total_timesteps=100000)

    def run(self, env):
        obs = env.reset()
        while True:
            obs = obs[np.newaxis, ...]
            action, n_state = self._model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                return info

    def save(self):
        self._model.save('agent.zip')

    @classmethod
    def load(cls, data, frame_bound, window_size):
        agent = Agent(data, frame_bound, window_size)
        agent._model = A2C.load('agent.zip')
        return agent
