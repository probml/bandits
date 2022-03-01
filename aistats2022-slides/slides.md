---
# try also 'default' to start simple
theme: default
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://images.unsplash.com/photo-1620460700571-320445215efb?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&h=1080&ixid=MnwxfDB8MXxyYW5kb218MHw5NDczNDU2Nnx8fHx8fHwxNjQ1MjY3NTc2&ixlib=rb-1.2.1&q=80&utm_campaign=api-credit&utm_medium=referral&utm_source=unsplash_source&w=1920
# apply any windi css classes to the current slide
---

# Subspace Neural Bandits
### AIStats 2022

Gerardo Duran-Martin, Queen Mary University of London, UK  
Aleyna Kara, Boğaziçi University, Turkey  
Kevin Murphy, Google Research, Brain Team

Feburary 2022

---

# Contextual bandits
### [Li, et.al. (2012)](https://arxiv.org/abs/1003.0146)

Let $\mathcal{A} = \{a^{(1)}, \ldots, a^{(K)}\}$ be a set of actions. At every time step $t=1,\ldots,T$
1. we are given a context ${\bf s}_t$ 
2. we decide, based on ${\bf s}_t$, an action $a_t \in \mathcal{A}$
3. we obtain a reward $r_t$ based on the context ${\bf s}_t$ and the chosen action $a_t$

<span style="background-color:#A7C7E76E">Our goal is to choose the set of actions that maximise the expected reward $\sum_{t=1}^T\mathbb{E}[R_t]$</span>.

---

# Thompson Sampling
<!-- One way to solve the bandit's problem -->
### [Agrawal and Goyal (2014)](https://arxiv.org/abs/1209.3352), [Russo, et.al. (2014)](https://arxiv.org/abs/1402.0298)
Let $\mathcal{D}_t = (s_t, a_t, r_t)$ be a sequence of observations. Let $\mathcal{D}_{1:t} = \{\mathcal{D}_1, \ldots, \mathcal{D}_t\}$.

At every time step $t=1,\ldots, T$, we follow the following procedure:
1. Sample $\boldsymbol\theta_t \sim p(\cdot \vert \mathcal{D}_{1:t})$
2. $a_t = \arg\max_{a \in \mathcal{A}} \mathbb{E}[R(s_t,a; \boldsymbol\theta_t)]$
3. Obtain $r_t \sim R(s_t,a_t; \boldsymbol\theta_t)$
4. Store $\mathcal{D}_t = (s_t, a_t, r_t)$

*Example*: Beta-Bernoulli bandit with $K=4$ arms.

<video width=900 loop="true" autoplay="autoplay" muted>
  <source src="/ts-bandits.mp4" type="video/mp4">
</video>


---

# Neural Bandits
### Characterising the reward function

Let $f: \mathcal{S}\times\mathcal{A}\times\mathbb{R}^D \to \mathbb{R}^K$ be a neural network. A neural bandit is a contextual bandit where the reward is taken to be

$$
  r_t \vert {\bf s}_t, a, \theta_t \sim \mathcal{N}\Big(f({\bf s}_t, a, \boldsymbol\theta_t), \sigma^2\Big)
$$


The main question: <span style="background-color:#A7C7E76E"> How to determine $\boldsymbol\theta_t$ at every time step $t$ using Thompson sampling?</span>  

We need to compute (or approximate) the posterior distribution of the parameters in the neural network:

$$
\begin{aligned}
  p(\boldsymbol\theta \vert \mathcal{D}_{1:t}) &= p(\boldsymbol\theta \vert \mathcal{D}_{1:t-1}, \mathcal{D}_t)\\
  &\propto p(\boldsymbol\theta \vert \mathcal{D}_{1:t-1}) p(\mathcal{D}_t \vert \boldsymbol\theta) \\
\end{aligned}
$$

---

# Subspace neural bandits
## Motivation

* Current state-of-the-art solutions, although efficient, are not fully Bayesian.
  1. Neural linear approximation
  2. Lim2 approximation
  3. Neural tangent approximation

<v-clicks>

* Fully Bayesian solutions are computationally expensive.
  <!-- Not an online method; very expensive to compute at every timestep -->
  1. Hamiltonian Monte Carlo (HMC) sampling of posterior beliefs
  <!-- Does not scale well as the number of parameters increases -->
  2. Extended Kalman Filter (EKF) online estimation of posterior beliefs
* We seek to solve the contextual-neural-bandit problem in a way that is **fully Bayesian** and **computationally-efficient**.

</v-clicks>

----

# Extended Kalman filter and neural networks
### [Singhal and Wu (1988)](https://proceedings.neurips.cc/paper/1988/hash/38b3eff8baf56627478ec76a704e9b52-Abstract.html)
Online learning of neural network parameters

$$
  \begin{aligned}
    \boldsymbol\theta_t \vert \boldsymbol\theta_{t-1} \sim \mathcal{N}(\boldsymbol\theta_{t-1}, \sigma^2 {\bf I}) \\
    r_t \vert \boldsymbol\theta_t \sim \mathcal{N}(f({\bf s}_t, a_t, \boldsymbol\theta_t), \sigma^2 {\bf I})
  \end{aligned}
$$

<br/>

<video class="horizontal-center" width=370 loop="true" autoplay="autoplay" muted>
  <source src="https://github.com/probml/probml-data/blob/main/data/ekf_mlp_demo.mp4?raw=true" type="video/mp4">
</video>

---

# Neural networks (in a subspace)
### [Li, et.al. (2018)](https://arxiv.org/abs/1804.08838), [Larsen, et.al. (2021)](https://arxiv.org/abs/2107.05802)
Stochastic gradient descent (SGD) for neural networks **in a subspace**.  
Neural networks live in a linear subspace.

$$
  \boldsymbol\theta({\bf z}_t) = {\bf A z}_{t} + \boldsymbol\theta_*
$$

<br>

<img class="horizontal-center" src="/subspace-neural-net.png" width=400>

---

# Our contribution: subspace neural bandits
### Extended Kalman filter and neural networks in a subspace.
We learn a subspace ${\bf z} \in \mathbb{R}^d$

$$
\begin{aligned}
  \boldsymbol\theta({\bf z}_t) &= {\bf A z}_{t} + \boldsymbol\theta_* \\
  {\bf z}_t \vert {\bf z}_{t-1}  &\sim \mathcal{N}({\bf z}_{t-1}, \tau^2 {\bf I}) \\
  {\bf r}_t \vert {\bf z}_t &\sim \mathcal{N}\Big(f({\bf s}_t, a_t, \boldsymbol\theta({\bf z}_t)), \sigma^2 {\bf I}\Big)
\end{aligned}
$$

<img class="horizontal-center" src="/subspace-neural-bandit-diagram.jpg" width=300 height=300>

---

# Results
### MNIST: cumulative reward  

Classification-turned-bandit problem.  
Maximum reward is $T=5000$ (total number of samples).

<br>
<br>

<img class="horizontal-center" width=500 src="/reward-bandits-mnist.png">

---

# Results
### MNIST: running time

<div class="centered">
  <img width=500 src="/time-bandits-mnist.png">
</div>

---

# Results
### Effect of subspace dimensionality
We seek to make $d$ as small as possible.

<br>

<img align="left" width=400 src="/covertype-snb-dim-reward.png">
<img align="right" width=400 src="/covertype-snb-dim-reward.png">

---

# Subspace neural bandits
### gerdm.github.io/aistats2022

<div class="centered">
  <h2 style="line-height:100px;"> <a href="https://arxiv.org/abs/2112.00195">📑 Paper</a></h2>
  <h2> <a href="https://github.com/probml/bandits">💻 Github repo</a></h2>
</div>
