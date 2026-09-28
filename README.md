Toy JAX kernel of Spectral Feedback (arXiv:2609.30456): Walsh-Hadamard edit-set value functions and test-time remask feedback on a length-8 absorbing-mask reverse process.

## Install

```
python -m venv .venv
.venv/bin/pip install -e ".[tests]"
```

## Run

```
.venv/bin/pytest
.venv/bin/python train.py
```

## What does not match the paper

No protein diffusion model, no ProxySPEX/Gurobi, no Megascale numbers; exact/truncated Walsh-Hadamard on L=8 and brute-force argmax stand in for sparse recovery + ILP.
