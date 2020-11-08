from environments.environment import AbstractEnvironment


class TaxiModelIrrelevanceAbstraction(AbstractEnvironment):
    def __init__(self):
        import gym
        self._env = gym.make("Taxi-v3").env
        self._abstract_env = gym.make("Taxi-v3").env
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._not_abstract_state = self._env.s

    def reset(self):
        """
          Resets the current state to the start state
        """
        self._env.reset()
        self._abstract_env.reset()

    def render(self):
        """
        Renders the environment.
        """
        self._abstract_env.render()

    def step(self, action):
        """
          Performs the given action in the current
          environment state and updates the environment.

          Returns (new_obs, reward, done, info)
        """
        new_obs, reward, done, info = self._env.step(action)
        self._not_abstract_state = new_obs
        taxi_row, taxi_col, pass_idx, dest_idx = self._env.decode(self._env.s)
        self._abstract_env.s = self._abstract_env.encode(0, taxi_col, pass_idx, dest_idx)
        return self._abstract_env.s, reward, done, info


class TaxiQPiIrrelevanceAbstraction(AbstractEnvironment):
    def __init__(self):
        import gym
        self._env = gym.make("Taxi-v3").env
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._not_abstract_state = self._env.s

    def reset(self):
        """
          Resets the current state to the start state
        """
        self._env.reset()

    def render(self):
        """
        Renders the environment.
        """
        self._env.render()

    def step(self, action):
        """
          Performs the given action in the current
          environment state and updates the environment.

          Returns (new_obs, reward, done, info)
        """
        abstract_new_obs, abstract_reward, abstract_done, abstract_info = self._abstract_step(action)
        new_obs, reward, done, info = self._env.step(action)
        self._not_abstract_state = new_obs
        return abstract_new_obs, reward, done, info

    def _abstract_step(self, action):
        taxi_row, taxi_col, pass_idx, dest_idx = self._env.decode(self._env.s)
        self._env.s = self._env.encode(0, taxi_col, pass_idx, dest_idx)
        abstract_new_obs, abstract_reward, abstract_done, abstract_info = self._env.step(action)
        self._env.s = self._not_abstract_state
        return abstract_new_obs, abstract_reward, abstract_done, abstract_info
