
import jax
import pandas as pd
from jax import numpy as jnp
from jax._src.typing import Array
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, OptimizeResult
from scipy import optimize
import numpy as onp
from numpy.typing import ArrayLike
from scipy.optimize import approx_fprime

from typing import List, Tuple
from jaxed.bandits.continuous_bandit import CBandit, optimize_powell, optimize_random
from jaxed.bandits.quantum_bandit import QuantumBandit, modulo_2pi, BPToyModel, QAOABandit
from jaxed.optimizers.lipschitz_best_arm import JL2_ND_LBAO

# very unelegant way of saving data
# Run optimizer from scipy.optimize.minimize
def run_gradient_optimizer(bandit: QuantumBandit, 
                      id, 
                      init_params: Array, 
                      constant_params: dict, 
                      target_minimum: float=0.0,
                      method="COBYLA",
                      qc="PQC"):
    
    def save_state(objective):
        method = "{}_{}".format(objective.method, objective.shots)
        plot_data = objective.plot_data
        for id, value in constant_params.items():
            plot_data[id] = objective.function_calls*[value]
        df = pd.DataFrame(plot_data)
        df["method"] = method
        df.to_csv("results/{}/results_{}_{}_{}.csv".format(qc, method, objective.qubits, objective.id))

    # run internally jitted function like in the wrapper
    def objective(x: ArrayLike):
        if not hasattr(objective, "key"):
            objective.key = jax.random.split(jax.random.PRNGKey(1234), num=(20,))[objective.id]
        if not hasattr(objective, "function_calls"):
            objective.function_calls = 0
        if not hasattr(objective, "plot_data"):
            objective.plot_data = {"minimum": [],
                "argmin": [],
                "samples": [],
                "true_expval": [], 
                "T": []}
        if not hasattr(objective, "converged"):
            objective.converged = False
        x = modulo_2pi(jnp.asarray(x))
        mean_jit = jax.jit(jnp.mean)
        y = mean_jit(bandit.reward(bandit.sample(objective.key, x)))
        # handle key optimally
        key2, key = jax.random.split(objective.key)
        objective.key = key 
        # update values
        true_expval = bandit.expected_value(jnp.asarray(x))
        if not objective.converged:
            for id, value in zip(["minimum", "argmin", "samples", "true_expval", "T"], 
                                    [y, 0, bandit.shots, true_expval, 1]):
                objective.plot_data[id].append(value)

            objective.function_calls += 1
            objective.converged = true_expval < target_minimum
        else:
            save_state(objective)
            # only way to make COBYLA stop
            raise StopIteration

        if objective.function_calls % 1000 == 0:
            print("Round ", objective.function_calls," Current minimum: ", y, " Current argmin: ", 0, 
                "qubits", constant_params["circuit_id"], "true expval: ", true_expval)
            save_state(objective)

        return y

    objective.id = id
    objective.qubits = bandit.wires
    objective.shots = bandit.shots
    objective.method = method
    objective.constant_params = constant_params
    
    res: OptimizeResult = minimize(objective, 
                                   method=method,
                                   x0=onp.array(init_params),
                                   tol=1e-8,
                                   options={"maxiter": onp.int64(1e9)})
    
    plot_data = objective.plot_data

    for id, value in constant_params.items():
        plot_data[id] = objective.function_calls*[value]

    return plot_data


# single VQA optimization run for given settings
def single_experiment(method, n_steps, qubits, id, shots, delta, L, qc):
    threshold = 0.05
    key = jax.random.PRNGKey(123456)
    key, key2 = jax.random.split(key)
    key = jax.random.split(key, num=(100,))[id]
    key2 = jax.random.split(key2, num=(100,))[id]
    shape_params = (6*qubits**2,) if qc=="PQC" else (2*qubits,)
    init_params = jax.random.uniform(key2, shape_params) 
    seeds = onp.random.randint(0, 1e10, size=(100,))
    bandit = quantum_to_cbandit(LineBandit, BPToyModel(qubits), init_params=init_params, midpoint=True) if qc=="PQC" \
            else quantum_to_cbandit(LineBandit, QAOABandit(qubits, qubits, 0.5, int(seeds[id])), init_params=init_params, midpoint=True)
    target_minimum = 0.4 if qc=="PQC" else 0.2

    if method == "BanditPowell":
        data = optimize_powell(bandit, key2, init_params, 
                                {"circuit_id": qubits, "run_id": id}, 
                                optimizer_cls=JL2_ND_LBAO,
                                n_steps=n_steps,
                                threshold=threshold,
                                target_minimum=target_minimum,
                                delta=delta,
                                L=L)
        method = "{}_{}".format(method, n_steps)
    elif method.split("_")[0] == "Random":
        data = optimize_random(bandit, key2, init_params, 
                                {"circuit_id": qubits, "run_id": id}, 
                                optimizer_cls=JL2_ND_LBAO,
                                n_steps=n_steps,
                                threshold=threshold,
                                target_minimum=target_minimum,
                                delta=delta,
                                L=L,
                                change_pr=method.split("_")[1])
        method = "{}_{}".format(method, n_steps)
    else:
        bandit = BPToyModel(qubits, shots=shots) if qc=="PQC" \
            else QAOABandit(qubits, qubits, 0.5, int(seeds[id]), shots=shots)
        data = run_gradient_optimizer(bandit,
                                        id,init_params, 
                                        {"circuit_id": qubits, "run_id": id},
                                        target_minimum=target_minimum,
                                        method=method,
                                        qc=qc)
        method = "{}_{}".format(method, shots)
    plot_data = data
    df = pd.DataFrame(plot_data)
    df["method"] = method
    df.to_csv("results/{}/results_{}_{}_{}_{}_{}.csv".format(qc, method, qubits, id, delta, L))
    return df

    
if __name__ == "__main__":
    from jaxed.bandits.quantum_bandit import BPToyModel, QAOABandit
    from jaxed.bandits.continuous_bandit import quantum_to_cbandit, OneDimBandit, LineBandit
    from argparse import ArgumentParser
    from jax import config

    # UNCOMMENT TO CACHE JIT-COMPILED FUNCTIONS

    #jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    #jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    #jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    #jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    config.update("jax_enable_x64", True)

    parser = ArgumentParser()
    parser.add_argument("--method",
                        default="Random_never",
                        type=str,
                        help="type of learning algorithm")
    parser.add_argument("--n_steps",
                        default=1,
                        type=str,
                        help="maximum depth")
    parser.add_argument("--qubits",
                        default=5,
                        type=str,
                        help="qubits of circuit")
    parser.add_argument("--id",
                        default=0,
                        type=str,
                        help="experiment id")
    parser.add_argument("--shots",
                        default=1,
                        type=str,
                        help="shots per quantum bandit call")
    parser.add_argument("--delta",
                        default=20,
                        type=str,
                        help="confidence parameter")
    parser.add_argument("--L",
                        default=0.25,
                        type=str,
                        help="Lipschitz constant")
    parser.add_argument("--qc",
                        default="QAOA",
                        type=str,
                        help="Quantum algorithm")
    
    args = parser.parse_args()
    method = args.method
    qubits = int(args.qubits)
    id = int(args.id)
    n_steps = int(args.n_steps)
    shots = int(args.shots)
    delta = float(args.delta)
    L = float(args.L)
    qc = args.qc
    single_experiment(method, n_steps, qubits, id, shots, delta, L, qc)
