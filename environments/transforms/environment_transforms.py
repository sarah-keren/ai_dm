from copy import deepcopy
from abc import ABC, abstractmethod
from itertools import groupby
from Cython.Utils import OrderedSet
from environments.environment import AbstractEnvironment


class TransformEnvironment(AbstractEnvironment):
    def __init__(self, is_it_custom_env, env_name, mapping_class=None):
        self._mapping_class = None
        if is_it_custom_env:
            pass
        else:
            import gym
            self._env = gym.make(env_name)
            if mapping_class:
                self._mapping_class = mapping_class(self._env)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self):
        """
          Resets the current state to the start state
        """
        return self._env.reset()

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
        cur_s = self._env.s
        new_obs, reward, done, info = self._env.step(action)
        if self._mapping_class:
            abstract_obs = self._mapping_class.mapping_step(cur_s, action)
            new_obs = abstract_obs
        return new_obs, reward, done, info


class MappingFunction(ABC):
    @abstractmethod
    def __init__(self, env):
        """
        Initialize the data structure of the mapping
        """
        self._env = deepcopy(env)

    @abstractmethod
    def mapping_step(self, state, action):
        """
        Step in the mapped environment
        """
        pass

    def _get_transition_info(self, state, action):
        transitions = self._env.P[state][action]
        if len(transitions) == 1:
            i = 0
        else:
            i = self._env.categorical_sample([t[0] for t in transitions], self._env.np_random)
        p, s, r, d = transitions[i]
        return p, s, r, d


class ModelIrrelevanceMapping(MappingFunction):
    def __init__(self, env):
        super().__init__(env)
        self._mapping_dict = {}
        self._init_mapping_dict()

    def mapping_step(self, state, action):
        prob, new_obs, reward, done = self._get_transition_info(state, action)
        abstract_new_obs = self._mapping_dict[new_obs]
        return abstract_new_obs

    def _init_mapping_dict(self):
        actions = self._env.action_space.n
        temp_mapping_dict = self._map_states_with_equal_rewards(actions)
        self._map_states_with_equal_probabilities(actions, temp_mapping_dict)

    def _map_states_with_equal_rewards(self, actions):
        temp_mapping_dict = {}
        states = len(self._env.P.keys())
        for s1 in range(states):
            if s1 not in temp_mapping_dict:
                temp_mapping_dict[s1] = OrderedSet([s1])
            for s2 in range(s1 + 1, states):
                if s2 not in temp_mapping_dict:
                    temp_mapping_dict[s2] = OrderedSet([s2])
                not_equal_rewards = False
                for a in range(actions):
                    if self._get_reward(s1, a) != self._get_reward(s2, a):
                        not_equal_rewards = True
                        break
                if not_equal_rewards:
                    continue
                temp_mapping_dict[s1].add(s2)
                temp_mapping_dict[s2].add(s1)
        return temp_mapping_dict

    def _map_states_with_equal_probabilities(self, actions, temp_mapping_dict):
        for s_set in temp_mapping_dict.values():
            set_probs = self._get_set_probabilities(s_set, actions)
            set_probs = dict(sorted(set_probs.items(), key=lambda x: x[1]))
            for k, group in groupby(set_probs.items(), key=lambda x: x[1]):
                for item in group:
                    group = list(group)
                    if len(group) > 0:
                        self._mapping_dict[item[0]] = group[0][0]

    def _get_reward(self, state, action):
        p, s, r, d = self._get_transition_info(state, action)
        return r

    def _get_set_probabilities(self, s_set, actions):
        set_probs = {}
        for s in s_set:
            in_set = 0
            for a in range(actions):
                next_s = self._get_next_s(s, a)
                if next_s in s_set:
                    in_set += 1 / actions
            set_probs[s] = in_set
        return set_probs

    def _get_next_s(self, state, action):
        p, s, r, d = self._get_transition_info(state, action)
        return s


class DimReductionMapping(MappingFunction):
    def __init__(self, env, reduction_idx=0):
        super().__init__(env)
        self.reduction_idx = reduction_idx
        self._mapping_dict = {}
        self._init_mapping_dict()

    def mapping_step(self, state, action):
        prob, new_obs, reward, done = self._get_transition_info(state, action)
        abstract_new_obs = self._mapping_dict[new_obs]
        return abstract_new_obs

    def _init_mapping_dict(self):
        states = len(self._env.P.keys())
        for s1 in range(states):
            if s1 not in self._mapping_dict:
                self._mapping_dict[s1] = OrderedSet([s1])
            for s2 in range(s1 + 1, states):
                if s2 not in self._mapping_dict:
                    self._mapping_dict[s2] = OrderedSet([s2])
                if self._the_same_abstract_state(s1, s2):
                    self._mapping_dict[s1].add(s2)
                    self._mapping_dict[s2].add(s1)
        self._mapping_dict = {k: v._set.pop() for k, v in self._mapping_dict.items()}

    def _the_same_abstract_state(self, s1, s2):
        state_1 = list(self._env.decode(s1))
        state_2 = list(self._env.decode(s2))
        state_idx = [i for i in range(len(state_1)) if i != self.reduction_idx]
        for idx in state_idx:
            if state_1[idx] != state_2[idx]:
                return False
        return True
