import jax.numpy as jnp
from jax import jacrev
from .base import NLDS


class DiagonalExtendedKalmanFilter(NLDS):
    """
    Implementation of the Diagonal Extended Kalman Filter for a nonlinear
    dynamical system with discrete observations. Also known as the
    Node-decoupled Extended Kalman Filter (NDEKF)
    """

    def __init__(self, fz, fx, Q, R):
        super().__init__(fz, fx, Q, R)
        self.Dfz = jacrev(fz)
        self.Dfx = jacrev(fx)

    @classmethod
    def from_base(cls, model):
        """
        Initialise class from an instance of the NLDS parent class
        """
        return cls(model.fz, model.fx, model.Q, model.R)

    def filter_step(self, state, xs, eps=0.001):
        """
        Run the Extended Kalman filter algorithm for a single step
        Paramters
        ---------
        state: tuple
            Mean, covariance at time t-1
        xs: tuple
            Target value and observations at time t
        """
        mu_t, Vt, t = state
        xt, obs = xs

        mu_t_cond = self.fz(mu_t)
        Ht = self.Dfx(mu_t_cond, *obs)

        Rt = self.R(mu_t_cond, *obs)
        xt_hat = self.fx(mu_t_cond, *obs)
        xi = xt - xt_hat
        A = jnp.linalg.inv(Rt + jnp.einsum("id,jd,d->ij", Ht, Ht, Vt))
        mu_t = mu_t_cond + jnp.einsum("s,is,ij,j->s", Vt, Ht, A, xi)
        Vt = Vt - jnp.einsum("s,is,ij,is,s->s", Vt, Ht, A, Ht, Vt) + self.Q(mu_t, t)

        return (mu_t, Vt, t + 1), (mu_t, None)

    def filter(self, init_state, sample_obs, observations=None, Vinit=None):
        """
        Run the Extended Kalman Filter algorithm over a set of observed samples.
        Parameters
        ----------
        init_state: array(state_size)
        sample_obs: array(nsamples, obs_size)
        Returns
        -------
        * array(nsamples, state_size)
            History of filtered mean terms
        * array(nsamples, state_size, state_size)
            History of filtered covariance terms
        """
        self.state_size, *_ = init_state.shape

        Vt = self.Q(init_state) if Vinit is None else Vinit

        t = 0
        state = (init_state, Vinit, t)
        observations = (observations,) if type(observations) is not tuple else observations
        xs = (sample_obs, observations)
        (mu_t, Vt, _), mu_t_hist = jax.lax.scan(self.filter_step, state, xs)

        return (mu_t, Vt), mu_t_hist
