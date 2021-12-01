from jax import vmap
from jax.lax import Precision
import jax.numpy as jnp
from jax.lax import scan
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class KalmanFilterNoiseEstimation:
    """
    Implementation of the Kalman Filtering and Smoothing
    procedure of a Linear Dynamical System with known parameters.
    This class exemplifies the use of Kalman Filtering assuming
    the model parameters are known.
    Parameters
    ----------
    A: array(state_size, state_size)
        Transition matrix
    C: array(observation_size, state_size)
        Observation matrix
    Q: array(state_size, state_size)
        Transition covariance matrix
    R: array(observation_size, observation_size)
        Observation covariance
    mu0: array(state_size)
        Mean of initial configuration
    Sigma0: array(state_size, state_size) or 0
        Covariance of initial configuration. If value is set
        to zero, the initial state will be completely determined
        by mu0
    timesteps: int
        Total number of steps to sample
    """

    def __init__(self, A, Q, mu0, Sigma0, v0, tau0, update_fn=None):
        self.A = A
        self.Q = Q
        self.mu0 = mu0
        self.Sigma0 = Sigma0
        self.v = v0
        self.tau = tau0
        self.__update_fn = update_fn

    def update(self, state, bel, *args):
        if self.__update_fn is None:
            return bel
        else:
            return self.__update_fn(state, bel, *args)

    def kalman_step(self, state, xt):
        mu, Sigma, v, tau = state
        x, y = xt

        mu_cond = jnp.matmul(self.A, mu, precision=Precision.HIGHEST)
        Sigmat_cond = jnp.matmul(jnp.matmul(self.A, Sigma, precision=Precision.HIGHEST), self.A,
                                 precision=Precision.HIGHEST) + self.Q

        e_k = y - x.T @ mu_cond
        s_k = x.T @ Sigmat_cond @ x + 1
        Kt = (Sigmat_cond @ x) / s_k

        mu = mu + e_k * Kt
        Sigma = Sigmat_cond - jnp.outer(Kt, Kt) * s_k

        v_update = v + 1
        tau = (v * tau + (e_k * e_k) / s_k) / v_update

        return mu, Sigma, v_update, tau

    def __kalman_filter(self, x_hist):
        """
        Compute the online version of the Kalman-Filter, i.e,
        the one-step-ahead prediction for the hidden state or the
        time update step
        Parameters
        ----------
        x_hist: array(timesteps, observation_size)
        Returns
        -------
        * array(timesteps, state_size):
            Filtered means mut
        * array(timesteps, state_size, state_size)
            Filtered covariances Sigmat
        * array(timesteps, state_size)
            Filtered conditional means mut|t-1
        * array(timesteps, state_size, state_size)
            Filtered conditional covariances Sigmat|t-1
        """
        _, (mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist) = scan(self.kalman_step,
                                                                       (self.mu0, self.Sigma0, 0), x_hist)
        return mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist

    def filter(self, x_hist):
        """
        Compute the online version of the Kalman-Filter, i.e,
        the one-step-ahead prediction for the hidden state or the
        time update step.
        Note that x_hist can optionally be of dimensionality two,
        This corresponds to different samples of the same underlying
        Linear Dynamical System
        Parameters
        ----------
        x_hist: array(n_samples?, timesteps, observation_size)
        Returns
        -------
        * array(n_samples?, timesteps, state_size):
            Filtered means mut
        * array(n_samples?, timesteps, state_size, state_size)
            Filtered covariances Sigmat
        * array(n_samples?, timesteps, state_size)
            Filtered conditional means mut|t-1
        * array(n_samples?, timesteps, state_size, state_size)
            Filtered conditional covariances Sigmat|t-1
        """
        has_one_sim = False
        if x_hist.ndim == 2:
            x_hist = x_hist[None, ...]
            has_one_sim = True
        kalman_map = vmap(self.__kalman_filter, 0)
        mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = kalman_map(x_hist)
        if has_one_sim:
            mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist = mu_hist[0, ...], Sigma_hist[0, ...], mu_cond_hist[
                0, ...], Sigma_cond_hist[0, ...]
        return mu_hist, Sigma_hist, mu_cond_hist, Sigma_cond_hist
