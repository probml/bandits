# Library of nonlinear dynamical systems
# Usage: Every discrete xKF class inherits from NLDS.
# There are two ways to use this library in the discrete case:
# 1) Explicitly initialize a discrete NLDS object with the desired parameters,
#    then pass it onto the xKF class of your choice.
# 2) Initialize the xKF object with the desired NLDS parameters using
#    the .from_base constructor.
# Way 1 is preferable whenever you want to use the same NLDS for multiple
# filtering processes. Way 2 is preferred whenever you want to use a single NLDS
# for a single filtering process

# Author: Gerardo Durán-Martín (@gerdm)

import jax
from jax.random import split, multivariate_normal


class NLDS:
    """
    Base class for the Nonliear dynamical systems' module
    """

    def __init__(self, fz, fx, Q, R):
        self.fz = fz
        self.fx = fx
        self.__Q = Q
        self.__R = R

    def Q(self, z, *args):
        if callable(self.__Q):
            return self.__Q(z, *args)
        else:
            return self.__Q

    def R(self, x, *args):
        if callable(self.__R):
            return self.__R(x, *args)
        else:
            return self.__R

    def __sample_step(self, input_vals, obs):
        key, state_t = input_vals
        key_system, key_obs, key = split(key, 3)

        state_t = multivariate_normal(key_system, self.fz(state_t), self.Q(state_t))
        obs_t = multivariate_normal(key_obs, self.fx(state_t, *obs), self.R(state_t, *obs))

        return (key, state_t), (state_t, obs_t)

    def sample(self, key, x0, nsteps, obs=None):
        """
        Sample discrete elements of a nonlinear system
        Parameters
        ----------
        key: jax.random.PRNGKey
        x0: array(state_size)
            Initial state of simulation
        nsteps: int
            Total number of steps to sample from the system
        obs: None, tuple of arrays
            Observed values to pass to fx and R
        Returns
        -------
        * array(nsamples, state_size)
            State-space values
        * array(nsamples, obs_size)
            Observed-space values
        """
        obs = () if obs is None else obs
        state_t = x0.copy()
        obs_t = self.fx(state_t)

        self.state_size, *_ = state_t.shape
        self.obs_t, *_ = obs_t.shape

        init_state = (key, state_t)
        _, hist = jax.lax.scan(self.__sample_step, init_state, obs, length=nsteps)

        return hist
