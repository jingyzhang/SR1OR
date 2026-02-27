# SR1OR
This is the official implementation of the paper "Selective Rank-1 Orthogonality Regularization: Rethinking Stability-Plasticity Balance in Continual Medical Image Classification". 

## Introduction

Faced with the emergence of new diseases in clinical practice, Continual Learning (CL) requires models to learn tasks sequentially with a critical balance between plasticity (new-task adaptation) and stability (prior-task retention). Pre-Trained Models (PTMs) offer a generalizable foundation for CL with compact parameter spaces injected incrementally, but limited by a stability-plasticity dilemma due to their uniform constraint on the whole space, e.g., full orthogonality or task-specific update. Hence, we introduce a principled {subspace decomposition} perspective to fundamentally revisit this dilemma, motivating a novel Selective Rank-1 Orthogonality Regularization (SR1OR) framework for stability-plasticity balance. Specifically, by exhaustively decomposing LoRA into unit rank-1 subspaces, we establish a Rank-1 Subspace Orientation Theory (R1SOT) that justifies unique orientation of each subspace based on subspace-local curvature and task-global sharpness. Guided by this, we develop Progressive Subspace Identification (PSI) to identify stability-oriented subspaces with large local curvature especially for tasks with high global sharpness. When new tasks arrive, we impose a Selective Orthogonality Constraint (SOC) only on these subspaces to mitigate interference for high stability, without regularization on residual subspaces to accommodate adaptation plasticity. Experiments on three class-incremental diagnosis benchmarks show a clear advantage for stability-plasticity balance than SOTA methods.

![fig1](https://github.com/jingyzhang/SR1OR/blob/main/figures/fig1.png?raw=true)


## Using the code:

The code is stable while using Python 3.10.0, CUDA >= 12.2

- Clone this repository:
```bash
git clone https://github.com/jingyzhang/SR1OR
cd SR1OR
```

To install all the dependencies :

```bash
conda env create sr1or python==3.10.0
conda activate sr1or
pip install -r requirements.txt
```
