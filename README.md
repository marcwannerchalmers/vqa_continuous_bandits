# Variational Quantum Algorithms with Continuous Bandits
Code for generating the results of the experiments in [Variational Quantum Algorithms with Continuous Bandits](https://arxiv.org/abs/2502.04021).

## Setup
Run the following command:
1. `python -m pip install - r requirements.txt`

## Usage
Example for possible setup:

```sh
python script.py --method "BanditPowell" --n_steps 5 --qubits 7 --id 2 --shots 1 --delta 0.1 --L 0.5 --qc "PQC"
```

## Arguments

### --method
- **Default:** `Random_never`
- **Type:** `str`
- **Description:** Specifies the type of learning algorithm to use. Options are `Random_always`, `Random_reject`, `Random_never`, `BanditPowell` and all suitable optimizers from SciPy.

### --n_steps
- **Default:** `1`
- **Type:** `str`
- **Description:** Defines the maximum depth $D_{\max}$ for the algorithm.

### --qubits
- **Default:** `5`
- **Type:** `str`
- **Description:** Sets the number of qubits $n$ in the quantum circuit.

### --id
- **Default:** `0`
- **Type:** `str`
- **Description:** Unique identifier for the experiment.

### --shots
- **Default:** `1`
- **Type:** `str`
- **Description:** Specifies the number of shots $N$ per quantum bandit call.

### --delta
- **Default:** `20`
- **Type:** `str`
- **Description:** Represents the confidence parameter $\delta$ used in the algorithm.

### --L
- **Default:** `0.25`
- **Type:** `str`
- **Description:** Defines the Lipschitz constant $L$ of the function to be optimized.

### --qc
- **Default:** `QAOA`
- **Type:** `str`
- **Description:** Specifies the quantum algorithm to be used (`PQC` or `QAOA`).


