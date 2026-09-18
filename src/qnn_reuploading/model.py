#pennyLane data-reuploading quantum neural network

from __future__ import annotations
from collections.abc import Sequence
from math import pi
import pennylane as qml
from pennylane import numpy as np

N_FEATURES = 11


#shallow variational circuit that repeatedly encodes 11 features
class DataReuploadingQNN:
    def __init__(self, n_qubits: int = 4, n_layers: int = 3, seed: int = 7):
        if n_qubits < 4 or n_qubits > 6:
            raise ValueError("n_qubits must be between 4 and 6")
        if n_layers < 1:
            raise ValueError("n_layers must be positive")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_schedule = tuple(
            (layer * n_qubits + wire) % N_FEATURES
            for layer in range(n_layers)
            for wire in range(n_qubits)
        )
        self.device = qml.device("default.qubit", wires=n_qubits, shots=None)
        self._qnode = qml.QNode(self._circuit, self.device, interface="autograd")
        rng = np.random.default_rng(seed)
        self.weights = np.array(
            0.05 * rng.standard_normal((n_layers, n_qubits, 2)),
            requires_grad=True,
        )

#encode one scheduled feature per wire, then apply trainable rotations
    def _circuit(self, features, weights):
        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                feature_index = (layer * self.n_qubits + wire) % N_FEATURES
                qml.RY(pi * features[feature_index], wires=wire)
                qml.RY(weights[layer, wire, 0], wires=wire)
                qml.RZ(weights[layer, wire, 1], wires=wire)
            for wire in range(self.n_qubits):
                qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])
        return qml.expval(qml.PauliZ(0))

#return a scalar expectation value in [-1, 1]
    def __call__(self, features, weights=None):
        values = np.asarray(features, requires_grad=False)
        if values.shape != (N_FEATURES,):
            raise ValueError(f"features must have shape ({N_FEATURES},)")
        parameters = self.weights if weights is None else weights
        return self._qnode(values, parameters)

#evaluate a sequence of feature vectors
    def batch_predict(self, feature_matrix: Sequence[Sequence[float]], weights=None):
        return np.array([self(row, weights=weights) for row in feature_matrix])

#mean squared error against targets in [-1, 1]
    def loss(self, feature_matrix, targets, weights=None):
        predictions = self.batch_predict(feature_matrix, weights=weights)
        target_values = np.asarray(targets)
        return np.mean((predictions - target_values) ** 2)
