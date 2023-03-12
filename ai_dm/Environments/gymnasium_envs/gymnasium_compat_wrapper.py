import gym


class GymnasiumCompatWrapper(gym.Wrapper):

    render_mode = None

    def reset(self, **kwargs):
        out = super().reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            if self.observation_space.contains(obs) and isinstance(info, dict):
                return obs, info

        return out, {}

    def step(self, action):
        out = super().step(action)
        if len(out) == 4:
            obs, reward, done, info = out
            trunc = False
        else:
            obs, reward, done, trunc, info = out
        return obs, reward, done, trunc, info

