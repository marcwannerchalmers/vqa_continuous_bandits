
from bandits import ContinuousBandit
from jaxed.bandits.continuous_bandit import CBandit
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import itertools
import pandas as pd
import time
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax import jit, vmap
from functools import partial
from optimizers.lipschitz_best_arm import L2_ND_LBAO, next_arms
from jaxopt import OSQP
from jax._src.typing import Array

# Optimized Gaussian confidence interval
def jgaussian_ci(bandit: CBandit, x, state: dict, length, alpha, key):
    n = int(math.ceil(2*jnp.log(2/alpha)/length**2))
    key_vec = jax.random.split(key, num=(n,))

    # sample midpoint of interval [x, x] for each key
    avg = jnp.mean(vmap(bandit.sample, (0, None, None), 0)(key_vec, state, [x,x]))

    return avg, n

# Optimized Bandit Optimizer
class JL2_ND_LBAO(L2_ND_LBAO):
    def __init__(self, bandit: CBandit, key, **kwargs):
        super().__init__(bandit, **kwargs)
        self.key = PRNGKey(1234) if key is None else key
        self.mins = []

    def step(self, state, t):
        t_start = time.time()
        arms = self.arms if t==1 else next_arms(self.arms, t, self.d, self.L)
        self.G.append(arms)

        means = np.zeros((len(arms)))
        length = self.ci_length(t)
        alpha = self.alpha(t)
        arms_keep = []
        pulls_per_arm_keep = []
        for i, h in enumerate(arms):
            self.key, key = jax.random.split(self.key)
            means[i], n_i = jgaussian_ci(self.bandit, h, state, length, alpha, key)
            self.pulls += n_i
            self.arms_pulled.append([h, t])
            self.pulls_per_arm.append(n_i)

        self.current_argmin = arms[np.argmin(means)]
        self.current_min = np.min(means)

        for i, h in enumerate(arms):
            if means[i] - length/2 - self.current_min <= 11/2**(t+4):
                arms_keep.append(h)
                pulls_per_arm_keep.append(self.pulls_per_arm[i])

        t_end = time.time()
        if self.save_location is not None:
            self.save_round(arms, means, self.pulls_per_arm, t_end-t_start)

        self.mins.append((self.current_argmin, self.current_min))
        self.arms = arms_keep
        self.pulls_per_arm = [] # save all of them separately
        return self.current_min, means







    
    