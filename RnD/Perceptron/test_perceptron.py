"""Behaviour checks: python -m unittest -v"""

import unittest

from perceptron import predict, train_perceptron


class PerceptronTests(unittest.TestCase):
    def setUp(self):
        self.X = [[0, 0], [0, 1], [1, 0], [1, 1]]

    def test_learns_linearly_separable_boolean_functions(self):
        for labels in ([0, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 0]):
            with self.subTest(labels=labels):
                weights, bias, history = train_perceptron(self.X, labels)
                self.assertEqual([predict(x, weights, bias) for x in self.X], labels)
                self.assertEqual(history[-1], 0)

    def test_xor_cannot_be_fit_and_training_is_bounded(self):
        labels = [0, 1, 1, 0]
        weights, bias, history = train_perceptron(self.X, labels, max_epochs=17)
        self.assertEqual(len(history), 17)
        self.assertNotIn(0, history)
        self.assertNotEqual([predict(x, weights, bias) for x in self.X], labels)

    def test_bias_can_learn_a_threshold_away_from_origin(self):
        X, labels = [[1], [2], [3], [4]], [0, 0, 1, 1]
        weights, bias, history = train_perceptron(X, labels)
        self.assertEqual([predict(x, weights, bias) for x in X], labels)
        self.assertNotEqual(bias, 0)
        self.assertEqual(history[-1], 0)

    def test_zero_score_predicts_one(self):
        self.assertEqual(predict([0, 0], [0, 0], 0), 1)

    def test_rejects_inconsistent_shapes_and_invalid_values(self):
        for X, y in (([], []), ([[]], [0]), ([[0], [1, 2]], [0, 1]),
                     ([[0]], []), ([[float("nan")]], [0]), ([[0]], [-1])):
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                train_perceptron(X, y)
        with self.assertRaises(ValueError):
            predict([1], [1, 2], 0)
        for rate in (0, -1, float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                train_perceptron(self.X, [0, 0, 0, 1], learning_rate=rate)
        for epochs in (0, 1.5, True):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                train_perceptron(self.X, [0, 0, 0, 1], max_epochs=epochs)


if __name__ == "__main__":
    unittest.main()
