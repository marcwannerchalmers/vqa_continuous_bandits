from typing import Any
import jax
from jax import numpy as jnp
from functools import partial
import numpy as np
from jax._src.typing import Array
from jax.lax import cond
from abc import ABC, abstractmethod
from jaxed.bandits.quantum_bandit import QuantumBandit

def _modulo_0_1(param: float):
    return param - jnp.floor(param)

def modulo_0_1(params: Array):
    return jax.vmap(_modulo_0_1)(params)

#base class for bandits
class Bandit(ABC):
    def __init__(self, n_arms=None) -> None:
        super().__init__()
        self.n_arms = n_arms

    # param: for discrete bandits, param is an integer, which 
    # refers to the arm index (starting at 0)
    # returns event omega
    @abstractmethod
    def sample(self, key, params):
        pass
    
    # returns reward X(omega)
    @abstractmethod
    def reward(self, outcome) -> float:
        pass

    @abstractmethod
    def outcome_to_idx(self, outcome: Array) -> int:
        pass
    
    @abstractmethod
    def expected_value(self) -> Array:
        pass

#only 1D at the moment
class CBandit(Bandit):
    def __init__(self, f, n_arms=None, midpoint=False, f_noise_free=None) -> None:
        super().__init__(n_arms)
        self.f_bandit = f
        self.f_inv = lambda key, x: 1.0 - f(key, x)
        self.f = f_noise_free
        self.inv = False
        self.interval_fn = (lambda key, params: (params[0] + params[1])/2) if midpoint \
            else (lambda key, params: jax.random.uniform(key, shape=params[0].shape,minval=params[0], maxval=params[1]))

    # params refers to the interval
    # need two keys
    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key, state, params):
        # split in three, such that we do not reproduce the split in the optimizer
        keys = jax.random.split(key, num=3)
        x = self.interval_fn(keys[0], params)
        return cond(self.inv, self.f_inv, self.f_bandit, keys[1], x)
    
    @partial(jax.jit, static_argnums=(0,))
    def reward(self, outcome) -> float:
        return outcome
    
    @partial(jax.jit, static_argnums=(0,))
    def outcome_to_idx(self, outcome: Array) -> int:
        return outcome.astype(jnp.int32)
    
    @partial(jax.jit, static_argnums=(0,))
    def expected_value(self, state, param) -> Array:
        f_nf_inv = lambda param: 1 - self.f_noise_free(param)
        return cond(self.inv, f_nf_inv, self.f_noise_free, param)
    
    def reverse(self):
        self.inv = True

    def __call__(self, x) -> Any:
        pass

# Wrapper class to sample from quantum bandit
def quantum_to_cbandit(bandit_cls, qbandit: QuantumBandit, **kwargs):
    def f_bandit(key, params):
        return qbandit.reward(qbandit.sample(key, qbandit.transform_unit_params(params)))
    def f_noise_free(params):
        return qbandit.expected_value(qbandit.transform_unit_params(params))
    return bandit_cls(f_bandit, f_noise_free=f_noise_free, **kwargs)

# Wrapper class to fix an index
class OneDimBandit(CBandit):
    def __init__(self, f, init_params, n_arms=None, midpoint=False, f_noise_free=None):
        super().__init__(f, n_arms, midpoint, f_noise_free)
        self.fixed_params = init_params
        self.variable_idx = 0

    def update_params(self, idx, fixed_params):
        self.variable_idx = idx
        self.fixed_params = fixed_params

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key, state, param: Array):
        param = jnp.array(param)
        params: Array = state["fixed_params"]
        params = jnp.stack([params for _ in range(2)])
        params = params.at[:, state["variable_idx"]].set(param[...,0])
        return super().sample(key, state, params)
    
    @partial(jax.jit, static_argnums=(0,))
    def expected_value(self, state, param) -> Array:
        f_nf_inv = lambda params: 1 - self.f(params)
        params: Array = state["fixed_params"]
        params = params.at[state["variable_idx"]].set(param)
        return cond(self.inv, f_nf_inv, self.f, params)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_params(self, state, param):
        return param
    
class LineBandit(CBandit):
    def __init__(self, f, init_params, n_arms=None, midpoint=False, f_noise_free=None):
        super().__init__(f, n_arms, midpoint, f_noise_free)
        self.fixed_params = init_params
        self.variable_idx = 0

    def update_params(self, idx, fixed_params):
        self.variable_idx = idx
        self.fixed_params = fixed_params

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key, state, param: Array):
        t = jnp.array(param)
        params = jnp.stack([state["fixed_params"] for _ in range(2)])
        params = modulo_0_1(params + jnp.outer(t, state["direction"]))
        return super().sample(key, state, params)
    
    @partial(jax.jit, static_argnums=(0,))
    def expected_value(self, state, param) -> Array:
        f_nf_inv = lambda params: 1 - self.f(params)
        params = modulo_0_1(state["fixed_params"] + param*state["direction"])
        return cond(self.inv, f_nf_inv, self.f, params)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_params(self, state, param):
        return modulo_0_1(state["fixed_params"] + param*state["direction"])
    
def optimization_iteration(bandit: LineBandit,
                           v: Array,
                           pr: Array,
                           key,
                           optimizer_cls,
                           delta,
                           L,
                           k,
                           threshold,
                           threshold_step,
                           n_steps,
                           constant_params,
                           target_minimum,
                           plot_data,
                           last_min):
    v_d = v
    v_d = v_d/jnp.linalg.norm(v_d)
    v_d = v_d/jnp.max(jnp.abs(v_d))
    key2, key = jax.random.split(key)
    state = {"direction":  v_d,
            "fixed_params": pr}
    key, key2 = jax.random.split(key)
    last_argmin = 0 # last argmin is 0, as pr = pr-1+t* * vn-1
    optimizer = optimizer_cls(bandit, delta=delta, L=L, key=key2, id=str(k), 
                                threshold=threshold, threshold_step=threshold_step)
    means, pulls = optimizer.run(state, n_steps=n_steps, last_min=last_min) # save_location=None to not save
    # compute current minimum, argmin
    argmins = jnp.array([m[0][0] for m in optimizer.mins]+[last_argmin])
    mins = jnp.array([m[1] for m in optimizer.mins]+[last_min])
    best_idx = jnp.argmin(mins)
    current_argmin = argmins[best_idx] # best lambda
    current_min = mins[best_idx]
     # update params
    pr = bandit.get_params(state, current_argmin)
    # print if there is an improvement
    true_expval = bandit.expected_value(state, current_argmin)
    if current_min < last_min:
        print("Round ", k," Current minimum: ", current_min, " Current argmin: ", current_argmin, 
                "qubits", constant_params["circuit_id"], "true expval: ", true_expval, "pulls", pulls)
        last_min = current_min
    # save data
    for id, value in zip(["minimum", "argmin", "samples", "true_expval", "T"], 
                        [current_min, current_argmin, pulls, true_expval, optimizer.t]):
        plot_data[id].append(value)
    stop = true_expval < target_minimum
    return key, plot_data, pr, stop, last_min
    
def optimize_powell(bandit: LineBandit, 
                    key, 
                    init_params: Array, 
                    constant_params: dict, 
                    optimizer_cls,
                    n_steps: int=2, 
                    threshold: float=0.05,
                    target_minimum: float=0.0,
                    delta: float=0.01,
                    L: float=1):
    last_min = 1
    pr = init_params
    p0 = pr
    d = pr.shape[0]
    plot_data = {"minimum": [],
                "argmin": [],
                "samples": [],
                "true_expval": [], 
                "T": []}
    k = 0
    threshold_step = 1
    directions = jnp.identity(d)
    true_expval = 1
    while true_expval > target_minimum:
        last_min = 1
        for i in range(d):
            v = directions[i]
            k += 1
            key, plot_data, pr, stop, last_min = optimization_iteration(bandit,
                                                                v,
                                                                pr,
                                                                key,
                                                                optimizer_cls,
                                                                delta,
                                                                L,
                                                                k,
                                                                threshold,
                                                                threshold_step,
                                                                n_steps,
                                                                constant_params,
                                                                target_minimum,
                                                                plot_data,
                                                                last_min)
            if stop:
                break    
        if stop:
            break
        # update directions
        pn = pr
        directions = jnp.roll(directions, -1, axis=0)
        directions = directions.at[d-1].set(pn - p0)
        # update p0
        print("new direction")
        key, plot_data, p0, stop, last_min = optimization_iteration(bandit,
                                                                directions[d-1],
                                                                pn,
                                                                key,
                                                                optimizer_cls,
                                                                delta,
                                                                L,
                                                                k,
                                                                threshold,
                                                                threshold_step,
                                                                n_steps,
                                                                constant_params,
                                                                target_minimum,
                                                                plot_data,
                                                                last_min)
        k += 1

    t = len(plot_data["argmin"])
    for id, value in constant_params.items():
        plot_data[id] = t*[value]

    return plot_data

def optimize_random(bandit: LineBandit, 
                    key, 
                    init_params: Array, 
                    constant_params: dict, 
                    optimizer_cls,
                    n_steps: int=2, 
                    threshold: float=0.05,
                    target_minimum: float=0.0,
                    delta: float=0.01,
                    L: float=1,
                    change_pr="always"):
    current_min = 0.5
    last_min = 1
    last_min_s = 1
    pr = init_params
    plot_data = {"minimum": [],
                "argmin": [],
                "samples": [],
                "true_expval": [], 
                "T": []}
    k = 0
    threshold_step = 1
    true_expval = 1
    while true_expval > target_minimum:
        if k%1000 == 0 and k > 0:
            print("Round ", k," Current minimum: ", last_min, " Current argmin: ", current_argmin, 
                      "qubits", constant_params["circuit_id"])
        key, key2 = jax.random.split(key)
        v = (0.5 - jax.random.uniform(key, (pr.shape[0],)))/2
        state = {"direction":  v,
                "fixed_params": pr}
        key, key2 = jax.random.split(key)
        optimizer = optimizer_cls(bandit, delta=delta, L=L,#*jnp.linalg.norm(v),
                                   key=key2, id=str(k), 
                                    threshold=threshold, threshold_step=threshold_step)
        means, pulls = optimizer.run(state, n_steps=n_steps, last_min=last_min) # save_location=None to not save
        key, key2 = jax.random.split(key)
        argmins = jnp.array([m[0][0] for m in optimizer.mins])
        mins = jnp.array([m[1] for m in optimizer.mins])
        best_idx = jnp.argmin(mins)
        current_argmin = argmins[best_idx]
        current_min = mins[best_idx]
        true_expval = bandit.expected_value(state, current_argmin) # have to take it where it changed the last time
        diff_current = last_min_s - current_min
        a = jax.random.uniform(key)
        key, key2 = jax.random.split(key)

        if change_pr == "always":
            pr = bandit.get_params(state, current_argmin)

        if a <= jnp.exp(400*(diff_current)) and change_pr == "reject":
            last_min_s = current_min
            pr = bandit.get_params(state, current_argmin)
            
        if current_min < last_min:
            pr = bandit.get_params(state, current_argmin)
            print("Round ", k," Current minimum: ", current_min, " Current argmin: ", current_argmin, 
                      "qubits", constant_params["circuit_id"], "pulls", pulls, "L", optimizer.L, "arms", len(optimizer.arms), "expval", true_expval)
            last_min = current_min

        for id, value in zip(["minimum", "argmin", "samples", "true_expval", "T"], 
                                [current_min, current_argmin, pulls, true_expval, optimizer.t]):
            plot_data[id].append(value)

        k += 1

    t = len(plot_data["argmin"])
    for id, value in constant_params.items():
        plot_data[id] = t*[value]

    return plot_data
