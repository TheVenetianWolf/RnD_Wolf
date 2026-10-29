"""A small, inspectable binary perceptron. Requires Python 3.10+, no packages.

Run: python perceptron.py
Labels are 0 and 1. A score of exactly zero predicts 1.
"""

from math import isfinite


def predict(x: list[float], weights: list[float], bias: float) -> int:
    """Predict one label; strict zip rejects a feature-count mismatch."""
    score = sum(
        w * value 
        for w, value in zip(weights, x, strict=True)
        ) + bias
    
    return int(score >= 0)


def train_perceptron(
    X: list[list[float]],
    y: list[int],
    *,
    learning_rate: float = 1.0,
    max_epochs: int = 100,
) -> tuple[list[float], float, list[int]]:
    """Return weights, bias, and the number of online mistakes in each epoch.

    An epoch is one complete pass through the training examples. We update
    immediately after each mistake, using a fixed order for reproducibility.
    A zero-mistake epoch confirms convergence on these training examples.
    Reaching max_epochs alone says nothing conclusive about separability.
    """
    if not X or not X[0] or len(X) != len(y):
        raise ValueError("Provide non-empty X and one label per example.")
    n_features = len(X[0])
    if any(len(row) != n_features for row in X):
        raise ValueError("Every example must have the same feature count.")
    if any(not isfinite(value) for row in X for value in row):
        raise ValueError("Features must be finite numbers.")
    if any(label not in (0, 1) for label in y):
        raise ValueError("Labels must be 0 or 1.")
    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if (
        isinstance(max_epochs, bool)
        or not isinstance(max_epochs, int)
        or max_epochs < 1
    ):
        raise ValueError("max_epochs must be a positive integer.")

    weights = [0.0] * n_features
    bias = 0.0
    mistakes_per_epoch = []

    for _ in range(max_epochs):
        mistakes = 0
        for x, target in zip(X, y, strict=True):
            prediction = predict(x, weights, bias)
            error = target - prediction

            if error != 0:
                for j in range(n_features):
                    weights[j] += learning_rate * error * x[j]
                bias += learning_rate * error
                mistakes += 1

        mistakes_per_epoch.append(mistakes)
        if mistakes == 0:
            break

    return weights, bias, mistakes_per_epoch


def main() -> None:
    # AND: only (1, 1) has label 1.
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 1]
    weights, bias, history = train_perceptron(X, y)

    print("Weights:", weights)
    print("Bias:", bias)
    print("Mistakes per epoch:", history)
    print("Zero-mistake epoch observed:", history[-1] == 0)
    for x, target in zip(X, y, strict=True):
        print(f"{x} -> predicted {predict(x, weights, bias)}, target {target}")

    # This is a check on the training examples, not an unseen-data evaluation.
    print("\nTry XOR by changing y to [0, 1, 1, 0].")


if __name__ == "__main__":
    main()
