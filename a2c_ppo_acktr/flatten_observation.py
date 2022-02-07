import gym.spaces as spaces

from gym import ObservationWrapper

from gym.spaces import Box, Discrete, Tuple, Dict, MultiBinary, MultiDiscrete

import numpy as np


def flatten(space, x):
    """Flatten a data point from a space.

    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.

    Accepts a space and a point from that space. Always returns a 1D array.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    """
    if isinstance(space, Box):
        return np.asarray(x, dtype=space.dtype).flatten()
    elif isinstance(space, Discrete):
        onehot = np.zeros(space.n, dtype=space.dtype)
        onehot[x] = 1
        return onehot
    elif isinstance(space, Tuple):
        return np.concatenate(
            [flatten(s, x_part) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, Dict):
        if all([isinstance(key, int) for key in x.keys()]):
            return np.concatenate(
                [flatten(s, x[0][key]) for key, s in space.spaces.items()])
        else:
            return np.concatenate(
                [flatten(s, x[key]) for key, s in space.spaces.items()])
    elif isinstance(space, MultiBinary):
        return np.asarray(x, dtype=space.dtype).flatten()
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x, dtype=space.dtype).flatten()
    else:
        raise NotImplementedError

class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return flatten(self.env.observation_space, observation)
