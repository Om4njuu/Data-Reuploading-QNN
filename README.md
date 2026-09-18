# Data-Reuploading QNN

A lightweight PennyLane implementation of a data-reuploading quantum neural network for 11 RDKit molecular features. The same 4-6 qubits repeatedly receive different feature slices across shallow layers, increasing the accessible Fourier frequency bound without increasing the qubit count.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Run the tests

```powershell
python -m pytest
```

## Run a small simulator training example

```powershell
qnn-train --qubits 4 --layers 3 --steps 30
```

The example uses four molecules, extracts 11 descriptors with RDKit, standardizes them, and trains the circuit against a small synthetic binary target. No cloud account or quantum hardware is required.

## Design

- `qnn_reuploading.features` - converts a SMILES string into 11 reproducible descriptors.
- `qnn_reuploading.model` - builds a shallow ring-entangled circuit and reuploads feature indices across layers.
- `qnn_reuploading.train` - runnable end-to-end simulator example.

The model accepts an input vector of exactly 11 values. With `n_qubits=4`, each layer consumes four scheduled feature values and three layers cover all 11 features. Additional layers intentionally re-expose the features, giving the variational circuit more nonlinear frequency capacity while keeping the qubit count fixed.
