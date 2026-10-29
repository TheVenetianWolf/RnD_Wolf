---
title: "Build and Understand a Perceptron"
date: 2026-10-29
tags:
  - learning/cs/machine-learning
  - feynman
  - cornell
status: "🌱 seedling"
confidence: low
related: []
---

# Build and Understand a Perceptron

> [!abstract] Summary
> A perceptron predicts one of two classes by weighting its input numbers, adding a bias, and checking whether the result reaches zero. Training changes its weights and bias whenever it makes a mistake. It can learn a straight boundary for separable training examples, but cannot represent XOR using its two raw inputs. Fitting training examples does not establish performance on unseen data.

## Start here

This first lesson assumes familiarity with Python lists, loops and functions. Use Python 3.10 or newer; no packages are needed for the model. Extract the archive into a folder and run `python perceptron.py` there. On systems using the `python3` command, substitute `python3`. Run `python -m unittest -v` for the five behaviour checks.

Read `predict` first, then the training loop in `train_perceptron`. The validation above the loop checks the input contract: consistent feature counts, finite numbers and labels of 0 or 1. We use a fixed example order and zero initial weights so results are reproducible. The diagram is an illustrative valid boundary, not the exact boundary returned by this run.

## Cornell Notes

### Cues & Questions

> [!question] What is a perceptron learning?
> A binary classifier chooses between two labels. A feature is one input number. An example is a list of features, such as `[1, 0]`. A target is the known correct label. For AND, the inputs `[0, 0]`, `[0, 1]`, `[1, 0]`, `[1, 1]` have targets `0, 0, 0, 1`.
>
> Training learns a weight for each feature and one bias. These are parameters: numbers adjusted using the training examples. The AND rule is not written into the prediction function; the targets teach it.

> [!question] How does prediction work?
> Calculate $z = w_1x_1 + w_2x_2 + b$. The weights $w_j$ control each feature's contribution. The bias $b$ shifts the threshold independently of the input. Predict $\hat y=1$ when $z\geq0$, otherwise predict $0$. A hat denotes a prediction, rather than the known target $y$.
>
> In code, `sum(w * value for w, value in zip(weights, x, strict=True))` calculates the weighted sum, also called a dot product. `strict=True` rejects unequal lengths instead of silently dropping features. `int(score >= 0)` converts `True` or `False` into `1` or `0`. The score is not a probability.

> [!question] How does a mistake change the model?
> Compute $e=y-\hat y$. Update every weight using $w_j\leftarrow w_j+\eta e x_j$ and update the bias using $b\leftarrow b+\eta e$. The learning rate $\eta$ is a positive number controlling update size; this lesson uses $1$.
>
> If the target is 1 but the prediction is 0, $e=+1$. For non-negative inputs, this raises the active features' weights and the bias, making a positive prediction easier. For a false positive, $e=-1$, so those quantities decrease. For negative-valued features the direction of each weight change also depends on the feature's sign. If the prediction is correct, $e=0$ and nothing changes.

> [!question] What is an epoch, and when do we stop?
> An epoch is one full pass through the training examples. This is online learning: each mistake changes the parameters immediately, before the next example. An epoch with no mistakes means the unchanged parameters correctly classified every training example.
>
> `mistakes_per_epoch` counts mistakes encountered during the pass, while parameters may be changing. It is not the number of mistakes the final model makes on the whole dataset. The counts need not decrease every epoch. A maximum epoch count prevents endless training. Reaching it does not prove the data are inseparable; the budget may simply be too small.

> [!question] Why does AND work, but XOR fail?
> The decision boundary is the set of inputs where $w_1x_1+w_2x_2+b=0$. In two dimensions this is a straight line when at least one weight is non-zero. AND can be separated by a line, for example $x_1+x_2-1.5=0$.
>
> XOR uses targets `0, 1, 1, 0`: diagonally opposite corners have the same label. No single straight line separates the two classes. On finite, strictly linearly separable training data, the perceptron with a fixed positive learning rate reaches a separating classifier after finitely many updates. That guarantee does not supply an arbitrary small epoch budget. More epochs cannot make a straight boundary represent XOR on its raw inputs.

> [!question] What do good research habits add?
> State the label and threshold conventions, bound the training loop, validate inputs, preserve reproducibility, and compare predictions with targets. Separate an algorithm check from an evaluation on unseen examples. Here, all four possible binary inputs form the complete AND truth table, so an arbitrary train/test split would obscure this first algorithm check. Later, with sampled real data, use a separate evaluation set and fit preprocessing on training data alone.
>
> Our fixed positive learning rate is a teaching choice. With zero initial parameters and this fixed example order, changing a constant positive learning rate merely scales all parameters in exact arithmetic; it does not change their predicted signs. Learning rates have different effects for other algorithms and initialisation schemes.

## Feynman Explanation

> [!tip] The Simple Version
> Imagine a decision made by adding up several weighted votes. Each input contributes a number multiplied by how much that input matters. A starting allowance, the bias, is added too. If the total reaches zero, say yes. When the answer is wrong, adjust the contributions and allowance, then try the next example.

### Analogies & Mental Models

**Weighted votes:** weights decide how strongly each feature contributes. A negative weight means a positive feature value counts against class 1.

**Starting allowance:** bias is like points added before counting any votes. It allows the dividing line to move away from the origin.

**Moving a ruler:** on a two-feature plot, training moves and rotates a straight dividing line to place the labelled examples on the correct sides.

### Where the Analogies Break Down

> [!warning] Limitations
> Features are numbers, not opinions, and may be negative or measured on different scales. The bias is a mathematical offset, not social bias. The ruler picture extends to a plane in three dimensions and a hyperplane in higher dimensions. A single perceptron still has a linear boundary in its input features and does not gain a curved boundary just by training longer.

### Knowledge gaps to close

1. **What happens at exactly zero?** This implementation predicts 1. Some textbook versions use labels −1 and +1 and update when a signed score is non-positive. Keep each version's labels, prediction rule and update rule together; do not copy only one formula between conventions.
2. **Why update the bias?** For `[0, 0]`, changing weights cannot change the score because both weighted inputs vanish. Updating the bias lets the model correct that example.
3. **Why does adding the input help?** A false negative updates $w$ by $\eta x$ and $b$ by $\eta$. Its own score increases by $\eta(\sum_jx_j^2+1)>0$. A false positive gets the same change with a negative sign. This helps that example, but may hurt others.
4. **Why not always get better each pass?** Corrections for different examples can interfere. The convergence guarantee for separable data does not require mistakes to decrease monotonically.

### A worked first update

> [!example] The first AND example
> Start with weights `[0, 0]`, bias `0`, and learning rate `1`. The first input `[0, 0]` has target `0`. Its score is zero, so the prediction is `1`. Therefore the error is `0 - 1 = -1`. Both weights stay zero because each input is zero, but the bias becomes `-1`.

The provided script finishes AND with weights `[2.0, 1.0]`, bias `-3.0`, and mistake history `[2, 3, 3, 2, 1, 0]`. Its last positive example has score exactly zero, which is classified as 1 by our convention. Many other parameters also implement AND; training does not find a unique answer.

## Connections

[[Linear Algebra]] describes the weighted sum as a dot product. [[Logistic Regression]] replaces the hard decision during modelling with a smooth probability model and a different training objective. [[Neural Networks]] combine units and non-linear transformations to represent more complex functions. [[Generalisation]] asks how learning transfers to unseen examples.

## Review Questions

1. What are a feature, a weight, a bias and a target?
2. Why does an all-zero input require a bias update to correct a false positive?
3. After the first worked update, what happens for input `[0, 1]`, target `0`?
4. With weights `[0, 0]`, bias `-1`, input `[1, 1]`, target `1`, and learning rate `1`, calculate the score, prediction, error and updated parameters.
5. Why does perfect AND training performance not establish performance on a separate real-world task?
6. Why can a larger epoch budget not make a single raw-input perceptron learn XOR exactly?

> [!success] Self-Test Protocol
> Answer each cue without reading its notes. Explain the prediction and correction rule in your own words, then trace one example by hand. Change the targets to OR (`[0, 1, 1, 1]`) and then XOR (`[0, 1, 1, 0]`), making a prediction about convergence before running the code. Revisit any uncertain section and update your learning status after review.

## Sources

[Cornell CS 4/5780: The Perceptron](https://courses.cis.cornell.edu/cs4780/2023sp/lectures/lecturenote03.html) explains the classifier, bias, geometry and convergence theorem using −1/+1 labels. This lesson uses the equivalent mistake-driven 0/1 update convention with an explicit tie rule. The code and observed outputs are supplied experiments, not copied from the source.

## Spaced Repetition Tracker

- [ ] Review in 1 day
- [ ] Review in 3 days
- [ ] Review in 7 days
- [ ] Review in 21 days
- [ ] Review in 60 days
