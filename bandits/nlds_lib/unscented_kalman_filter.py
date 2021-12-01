import jax.numpy as jnp
from jax.random import split, choice, multivariate_normal
from jax.lax import scan
from jax.ops import index_update
from jax.scipy import stats

from .base import NLDS


class UnscentedKalmanFilter(NLDS):
    """
    Implementation of the Unscented Kalman Filter for discrete time systems
    """

    def __init__(self, fz, fx, Q, R, alpha, beta, kappa, d):
        super().__init__(fz, fx, Q, R)
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lmbda = alpha ** 2 * (self.d + kappa) - self.d
        self.gamma = jnp.sqrt(self.d + self.lmbda)

    @classmethod
    def from_base(cls, model, alpha, beta, kappa, d):
        """
        Initialise class from an instance of the NLDS parent class
        """
        return cls(model.fz, model.fx, model.Q, model.R, alpha, beta, kappa, d)

    @staticmethod
    def sqrtm(M):
        """
        Compute the matrix square-root of a hermitian
        matrix M. i,e, R such that RR = M
        
        Parameters
        ----------
        M: array(m, m)
            Hermitian matrix
        
        Returns
        -------
        array(m, m): square-root matrix
        """
        evals, evecs = jnp.linalg.eigh(M)
        R = evecs @ jnp.sqrt(jnp.diag(evals)) @ jnp.linalg.inv(evecs)
        return R

    def filter(self, init_state, sample_obs, observations=None, Vinit=None):
        """
        Run the Unscented Kalman Filter algorithm over a set of observed samples.
        Parameters
        ----------
        sample_obs: array(nsamples, obs_size)
        Returns
        -------
        * array(nsamples, state_size)
            History of filtered mean terms
        * array(nsamples, state_size, state_size)
            History of filtered covariance terms
        """
        wm_vec = jnp.array([1 / (2 * (self.d + self.lmbda)) if i > 0
                            else self.lmbda / (self.d + self.lmbda)
                            for i in range(2 * self.d + 1)])
        wc_vec = jnp.array([1 / (2 * (self.d + self.lmbda)) if i > 0
                            else self.lmbda / (self.d + self.lmbda) + (1 - self.alpha ** 2 + self.beta)
                            for i in range(2 * self.d + 1)])
        nsteps, *_ = sample_obs.shape
        mu_t = init_state
        Sigma_t = self.Q(init_state) if Vinit is None else Vinit
        if observations is None:
            observations = [()] * nsteps
        else:
            observations = [(obs,) for obs in observations]

        mu_hist = jnp.zeros((nsteps, self.d))
        Sigma_hist = jnp.zeros((nsteps, self.d, self.d))

        mu_hist = index_update(mu_hist, 0, mu_t)
        Sigma_hist = index_update(Sigma_hist, 0, Sigma_t)

        for t in range(nsteps):
            # TO-DO: use jax.scipy.linalg.sqrtm when it gets added to lib
            comp1 = mu_t[:, None] + self.gamma * self.sqrtm(Sigma_t)
            comp2 = mu_t[:, None] - self.gamma * self.sqrtm(Sigma_t)
            # sigma_points = jnp.c_[mu_t, comp1, comp2]
            sigma_points = jnp.concatenate((mu_t[:, None], comp1, comp2), axis=1)

            z_bar = self.fz(sigma_points)
            mu_bar = z_bar @ wm_vec
            Sigma_bar = (z_bar - mu_bar[:, None])
            Sigma_bar = jnp.einsum("i,ji,ki->jk", wc_vec, Sigma_bar, Sigma_bar) + self.Q(mu_t)

            Sigma_bar_half = self.sqrtm(Sigma_bar)
            comp1 = mu_bar[:, None] + self.gamma * Sigma_bar_half
            comp2 = mu_bar[:, None] - self.gamma * Sigma_bar_half
            # sigma_points = jnp.c_[mu_bar, comp1, comp2]
            sigma_points = jnp.concatenate((mu_bar[:, None], comp1, comp2), axis=1)

            x_bar = self.fx(sigma_points, *observations[t])
            x_hat = x_bar @ wm_vec
            St = x_bar - x_hat[:, None]
            St = jnp.einsum("i,ji,ki->jk", wc_vec, St, St) + self.R(mu_t, *observations[t])

            mu_hat_component = z_bar - mu_bar[:, None]
            x_hat_component = x_bar - x_hat[:, None]
            Sigma_bar_y = jnp.einsum("i,ji,ki->jk", wc_vec, mu_hat_component, x_hat_component)
            Kt = Sigma_bar_y @ jnp.linalg.inv(St)

            mu_t = mu_bar + Kt @ (sample_obs[t] - x_hat)
            Sigma_t = Sigma_bar - Kt @ St @ Kt.T

            mu_hist = index_update(mu_hist, t, mu_t)
            Sigma_hist = index_update(Sigma_hist, t, Sigma_t)

        return mu_hist, Sigma_hist

    def __init__(self, fz, fx, Q, R):
        """
        Implementation of the Bootrstrap Filter for discrete time systems
        **This implementation considers the case of multivariate normals**
        to-do: extend to general case
        """
        super().__init__(fz, fx, Q, R)

    def __filter_step(self, state, obs_t):
        nsamples = self.nsamples
        indices = jnp.arange(nsamples)
        zt_rvs, key_t = state

        key_t, key_reindex, key_next = split(key_t, 3)
        # 1. Draw new points from the dynamic model
        zt_rvs = multivariate_normal(key_t, self.fz(zt_rvs), self.Q(zt_rvs))

        # 2. Calculate unnormalised weights
        xt_rvs = self.fx(zt_rvs)
        weights_t = stats.multivariate_normal.pdf(obs_t, xt_rvs, self.R(zt_rvs, obs_t))

        # 3. Resampling
        pi = choice(key_reindex, indices,
                    p=weights_t, shape=(nsamples,))
        zt_rvs = zt_rvs[pi, ...]
        weights_t = jnp.ones(nsamples) / nsamples

        # 4. Compute latent-state estimate,
        #    Set next covariance state matrix
        mu_t = jnp.einsum("im,i->m", zt_rvs, weights_t)

        return (zt_rvs, key_next), mu_t

    def filter(self, key, init_state, sample_obs, nsamples=2000, Vinit=None):
        """
        init_state: array(state_size,)
            Initial state estimate
        sample_obs: array(nsamples, obs_size)
            Samples of the observations
        """
        m, *_ = init_state.shape
        nsteps = sample_obs.shape[0]
        mu_hist = jnp.zeros((nsteps, m))

        key, key_init = split(key, 2)
        V = self.Q(init_state) if Vinit is None else Vinit
        zt_rvs = multivariate_normal(key_init, init_state, V, shape=(nsamples,))

        init_state = (zt_rvs, key)
        self.nsamples = nsamples
        _, mu_hist = scan(self.__filter_step, init_state, sample_obs)

        return mu_hist
