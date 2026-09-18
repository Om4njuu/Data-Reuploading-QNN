#small local simulator training example

from __future__ import annotations
import argparse
from pennylane import numpy as np
from .features import featurize_smiles, standardize
from .model import DataReuploadingQNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=4, choices=(4, 5, 6))
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN(CC)CC"]
    features = np.array(standardize([featurize_smiles(item) for item in smiles]))
    targets = np.array([-1.0, 1.0, -1.0, 1.0])
    model = DataReuploadingQNN(args.qubits, args.layers)
    weights = model.weights
    optimizer = __import__("pennylane").GradientDescentOptimizer(stepsize=0.15)

    for step in range(args.steps):
        weights, loss = optimizer.step_and_cost(lambda current: model.loss(features, targets, current), weights)
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step={step + 1:03d} loss={float(loss):.6f}")

    predictions = model.batch_predict(features, weights)
    print("predictions:", [round(float(value), 3) for value in predictions])


if __name__ == "__main__":
    main()
