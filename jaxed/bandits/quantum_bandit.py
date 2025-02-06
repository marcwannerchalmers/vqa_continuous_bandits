import jax
from jax import numpy as np
import pennylane as qml
from abc import abstractmethod
from functools import partial
import math
import networkx as nx

import numpy as onp
from numpy.typing import ArrayLike
from scipy.sparse.linalg import eigsh

from jaxed.bandits.bandit import Bandit
from jax._src.typing import Array
from typing import Optional

#base class for quantum bandits
#wires: dimension of quantum system 
#meas_fun: Pennylane function for measurement, e.g. sample, expval,etc.
class QuantumBandit(Bandit):
    def __init__(self, wires, meas_fun, shots=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.meas_fun = meas_fun
        self.wires = wires
        self.shots = shots
    
    @partial(jax.jit, static_argnums=(0,)) #ignore "self" in jit. Only needs to be compiled once per bandit (at most)
    def sample(self, key, params):
        dev = qml.device('default.qubit', wires=self.wires, shots=self.shots, seed=key)

        # note that jax does not support measuring arbitrary quantum ops (e.g. PauliX)
        # we circumvent this by writing O = U*Z and measuring in Z-basis
        # may need to scale outcomes non-Pauli Observables  
        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit():
            self.state()
            self.measurement(params)
            return [self.meas_fun(qml.PauliZ(wires=i)) for i in range(self.wires)]

        return self.adjust_dim(circuit())

    # it seems we need to have a function, which returns the state every time
    # in order for 
    @abstractmethod
    def state(self) -> qml.operation.Operation:
        pass
    
    # part corresponding to 'U' mentioned above
    @abstractmethod
    def measurement(self, params) -> qml.operation.Operation:
        pass

    #current workaround for 1-d, as we need array for outcome to index
    def adjust_dim(self, x):
        return np.array(x)

    # map +-1 vector to integer indices
    @partial(jax.jit, static_argnums=(0,))
    def outcome_to_idx(self, outcome: Array) -> int:
        outcome = ((outcome + 1)/2)
        def body_fun(i, power_res):
            power, res = power_res
            return (2*power, res + power*outcome[i])
        
        power_res = jax.lax.fori_loop(0, outcome.shape[0], body_fun, (1, 0))
        return power_res[1].astype(int)
    
    # TODO: version of this with parameters
    def get_expectations(self):
        dev = qml.device("default.qubit", wires=self.wires)

        # note that jax does not support measuring arbitrary quantum ops (e.g. PauliX)
        # we circumvent this by writing O = U*Z and measuring in Z-basis
        # may need to scale outcomes non-Pauli Observables  
        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit(param):
            self.state()
            self.measurement(param)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.wires)]
        
        expectations = []
        for arm in range(self.n_arms):
            expectations.append(self.reward(self.adjust_dim(circuit(arm))))

        return np.array(expectations)
    
    @partial(jax.jit, static_argnums=(0,))
    def expected_value(self, params):
        dev = qml.device("default.qubit", wires=self.wires)

        # note that jax does not support measuring arbitrary quantum ops (e.g. PauliX)
        # we circumvent this by writing O = U*Z and measuring in Z-basis
        # may need to scale outcomes non-Pauli Observables  
        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit():
            self.state()
            self.measurement(params)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.wires)]
        expectation = self.reward(self.adjust_dim(circuit()))

        return expectation
    
    """@partial(jax.jit, static_argnums=(0,))
    def outcome_to_scalar(self, outcome: Array):
        raise NotImplementedError()"""
    
    def plot_circuit(self, params):
        def circuit(params):
            self.state()
            self.measurement(params)
            return [self.meas_fun(qml.PauliZ(wires=i)) for i in range(self.wires)]
        drawer = qml.draw(circuit)
        print(drawer(params))

    # overwrite if parameter space is not in unit interval
    def transform_unit_params(self, params):
        return params

def _modulo_2pi(param: float):
    return param - np.floor(param/(2 * np.pi)) * 2 * np.pi 

def modulo_2pi(params: Array):
    return jax.vmap(_modulo_2pi)(params)


# Bandit with two observables as arms, Z and X
# state_params: 2-tuple containing rotation angles for Z- and Y-rotation
# can be more or less copy pasted to make general single qubit bandit
class ZXBandit(QuantumBandit):
    def __init__(self, state_params, **kwargs) -> None:
        wires = 1
        meas_fun = qml.sample
        super().__init__(wires, meas_fun, **kwargs)
        self.ops = np.array([0, 1])
        self.state_params = state_params
        self.n_arms = 2
    
    def state(self) -> qml.operation.Operation:
        return qml.prod(qml.RZ(self.state_params[0],0), qml.RY(self.state_params[1], 0))

    def measurement(self, param) -> qml.operation.Operation:
        rot = self.ops[param]
        return qml.prod(qml.RX(rot*np.pi, 0), qml.RY(rot*np.pi/2, 0))
    
    def reward(self, outcome) -> float:
        return outcome[0]


#input: wires x 2 array with RY, RZ angles

def get_random_state_params(wires, key):
    return jax.random.uniform(key, (wires, 2))

#(2*n_arms)**wires arms for uniform grid
def get_arms(wires, n_arms, mode, key=None):
    arms = None
    if mode=="random":
        arms = jax.random.uniform(key, (n_arms, wires, 2))
    elif mode=="uniform_grid":
        arms_single_param = np.linspace(0, 1, n_arms, endpoint=False)
        arms_single_param_all = np.array(np.meshgrid(*[arms_single_param 
                                                       for _ in range(wires)])).T.reshape(-1, wires)
        angles1 = np.repeat(arms_single_param_all, 2, axis=0)
        angles2 = np.tile(arms_single_param_all, (2, 1))
        arms = np.stack(angles1, angles2, axis=2)
    return arms

# wires: number of wires
# state_params: params ([0,1]x[0,1])**wires to determine the state 
# n_arms: number of arms
# arms: concrete 2d array with columns arm parameters
# Either n_arms & mode or arms have to be not None
# If arms is not None, it will be used and n_arms, mode ignored
# mode: "random": n_arms random arm states
#       "uniform_grid": uniform grid, such that we obtain (2*n_arms)**wires arms
class MultiQubitProductBandit(QuantumBandit):
    def __init__(self, wires, state_params, n_arms=None, 
                 arms=None, mode=None, key=None) -> None:
        assert n_arms is not None or arms is not None
        assert n_arms is None or mode is not None
        self.arms = get_arms(wires, n_arms, mode, key) if arms is None else arms
        n_arms = len(self.arms)
        super().__init__(wires, qml.sample, n_arms=n_arms)
        self.state_params = state_params
   
    def state(self) -> qml.operation.Operation:
        ops = [qml.prod(qml.RZ(op[0]*2*np.pi, i), qml.RY(op[1]*np.pi, i)) 
               for i, op in enumerate(self.state_params)] 
        return ops

    def measurement(self, arm_idx) -> qml.operation.Operation:
        ops = [qml.prod(qml.RY(-op[1]*np.pi, i), qml.RZ(-op[0]*2*np.pi, i)) 
               for i, op in enumerate(self.arms[arm_idx])] 
        return ops
    
    def reward(self, outcome) -> float:
        return np.sum(outcome)
    
class CascadingCZBandit(QuantumBandit):
    def __init__(self, wires, num_layers, K_matrix, 
                 meas_indices=[0,1], **kwargs):
        meas_fun = qml.sample
        super().__init__(wires, meas_fun, **kwargs)
        self.meas_inidces = meas_indices
        self.K_matrix = K_matrix
        self.num_layers = num_layers

    def state(self) -> qml.operation.Operation:
        # TODO: Check if exp(-1(\pi/8) \sigma_Y) is actually R_Y
        ops = [qml.RY(np.pi/8, i)
               for i in range(self.wires)]
        return ops
    
    def cascade_layer(self, param_vector: Array):
        ops_cascade = [qml.CZ(wires=[i, i+1]) 
                       for i in range(self.wires - 1)] 
        ops_rotation = [qml.prod(qml.RX(param_vector[i], i), qml.RY(param_vector[i], i),
                                 qml.RZ(param_vector[i], i)) for i in range(self.wires)]

        ops = ops_cascade
        ops.extend(ops_rotation)
        return ops

    # here, params has to be shaped as nxl matrix, where l is the number of layers
    def measurement(self, params):
        ops = []
        for l in range(self.num_layers):
            ops.extend(self.cascade_layer(params[l,:]))
        return ops
    
    @partial(jax.jit, static_argnums=(0,))
    def reward(self, outcome):
        return np.product(np.take(outcome, self.meas_inidces, unique_indices=True))

# PQC model
class BPToyModel(QuantumBandit):
    def __init__(self, wires, **kwargs):
        meas_fun = qml.sample
        super().__init__(wires, meas_fun, **kwargs)
    
    def U(self, params: Array, wire):
        return qml.prod(qml.RZ(params[1] + np.pi, wire), qml.RX(np.pi/2, wire),
                        qml.RZ(params[0] + np.pi, wire), qml.RZ(params[2], wire))
    
    def layer(self, params):
        ops = [*[self.U(params[3*i:3*(i+1)], i) for i in range(self.wires)],
               *[qml.CNOT([2*i, 2*i+1]) for i in range(int(self.wires/2))], # CNOT from even to odd
               *[self.U(params[3*i:3*(i+1)], i-self.wires) for i in range(self.wires,2 * self.wires)],
               *[qml.CNOT([(2*i - 1)%self.wires, 2*i]) for i in range(math.ceil(self.wires/2))]] # CNOT from odd to even
        return ops
    
    def measurement(self, params: Array):
        ops = []
        for i in range(self.wires):
            ops.extend(self.layer(params[i*(6*self.wires):(i+1)*(6*self.wires)]))

        return ops
    
    def state(self):
        return []
    
    def plot_layer(self, params):
        def circuit(params):
            for i in range(self.wires):
                self.layer(params[i*(6*self.wires):(i+1)*(6*self.wires)])
            return [self.meas_fun(qml.PauliZ(wires=i)) for i in range(self.wires)]
        drawer = qml.draw(circuit)
        print(drawer(params))
    
    # 1 - rate of outcomes, that are 0
    @partial(jax.jit, static_argnums=(0,))
    def reward(self, outcome: Array):
        rate_zeros = np.mean(outcome + np.ones_like(outcome))/2
        return 1 - rate_zeros
    
    def transform_unit_params(self, params):
        return params * 2 * np.pi

def generate_connected_graph(n: int, p: float, seed: Optional[int] = None) -> nx.Graph:
    """Generate a connected Erdos-Renyi graph."""
    while True:
        graph = nx.erdos_renyi_graph(n, p, seed=seed)
        if nx.is_connected(graph):
            break
        if seed is not None:
            seed = hash(31*seed+17)
    return graph

def compute_maxcut_value(graph: nx.Graph, x: ArrayLike):
    return sum([x[i] + x[j] - 2*x[i]*x[j] for i,j in graph.edges])

def compute_maxcut_optimum(graph: nx.Graph, cost_h: qml.Hamiltonian):
    cost_h_sparse = cost_h.sparse_matrix().astype(np.float64)
    w, v = eigsh(cost_h_sparse, k=2, which='BE')
    n = graph.number_of_nodes()
    max_val = -w[0]
    min_val = -w[1]
    return min_val, max_val

# QAOA circuit model
class QAOABandit(QuantumBandit):
    def __init__(self, wires, n_layers, p, seed, **kwargs):
        meas_fun = qml.sample
        super().__init__(wires, meas_fun, **kwargs)
        self.graph = generate_connected_graph(wires, p, seed)
        self.n_layers = n_layers
        self.cost_h, self.mixer_h = qml.qaoa.maxcut(self.graph)
        self.min_val, self.max_val = compute_maxcut_optimum(self.graph, self.cost_h)

    def qaoa_layer(self, gamma, alpha):
        qml.qaoa.cost_layer(gamma, self.cost_h)
        qml.qaoa.mixer_layer(alpha, self.mixer_h)

    def measurement(self, params: Array):
        return qml.layer(self.qaoa_layer, self.n_layers, params[::2], params[1::2])
  
    def state(self):
        ops = []
        for w in range(self.wires):
            ops.append(qml.Hadamard(wires=w))
        return ops
    
    # computes 1 - approximation ratio of cut corresponding to outcome
    @partial(jax.jit, static_argnums=(0,))
    def reward(self, outcome):
        # turn into 0, 1 array
        outcome = np.abs((outcome - 1)//2)
        val = compute_maxcut_value(self.graph, outcome)
        return 1 - (val - self.min_val)/(self.max_val - self.min_val)
    
    @partial(jax.jit, static_argnums=(0,))
    def expected_value(self, params):
        key = jax.random.PRNGKey(123456)
        dev = qml.device('default.qubit', wires=self.wires, shots=100000, seed=key)

        # note that jax does not support measuring arbitrary quantum ops (e.g. PauliX)
        # we circumvent this by writing O = U*Z and measuring in Z-basis
        # may need to scale outcomes non-Pauli Observables  
        @qml.qnode(dev, interface="jax", diff_method=None)
        def circuit():
            self.state()
            self.measurement(params)
            return [self.meas_fun(qml.PauliZ(wires=i)) for i in range(self.wires)]

        outcomes = self.adjust_dim(circuit())
        expectation = np.mean(self.reward(outcomes))

        return expectation
    
    def plot_layer(self, params):
        def circuit(params):
            gamma, alpha = params
            self.qaoa_layer(gamma, alpha)
            return [self.meas_fun(qml.PauliZ(wires=i)) for i in range(self.wires)]
        drawer = qml.draw(circuit)
        print(drawer(params))
    
    def transform_unit_params(self, params):
        return params * 2 * np.pi