# ML Lab 401 — Support Vector Machines

## Dataset

**Dataset:** Fashion-MNIST (subset of 2000 samples)  
**Training set:** 1800 samples  
**Test set:** 200 samples  

The Fashion-MNIST dataset contains grayscale images of clothing items belonging to **10 classes**:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

All models were implemented using **NumPy** and the quadratic programming solver **CVXOPT**.

---

# Task 1 — SVM Binary Classifier with Non-Linear Kernels

The Support Vector Machine was implemented by solving the **dual optimization problem** described in *Bishop (Equations 7.32–7.34)*.

Let

$$
T = \mathrm{diag}(t)
$$

be the diagonal matrix of labels and \(K\) the kernel matrix.

The dual optimization problem becomes

$$
\min_{\alpha}
\frac{1}{2}\alpha^T (TKT)\alpha - \mathbf{1}^T \alpha
$$

subject to the following constraints.

---

## Box Constraints

Each Lagrange multiplier must satisfy

$$
0 \le \alpha_i \le C
$$

This was encoded for the QP solver as

$$
G =
\begin{bmatrix}
I \\
- I
\end{bmatrix},
\quad
h =
\begin{bmatrix}
C\mathbf{1} \\
0
\end{bmatrix}
$$

---

## Equality Constraint

The dual formulation also requires

$$
t^T \alpha = 0
$$

This ensures the resulting separating hyperplane satisfies the optimality conditions of the SVM.

---

## Support Vector Identification

After solving the quadratic program, support vectors were identified as samples with

$$
\alpha_i > 10^{-5}
$$

Only these samples influence the final decision boundary.

---

## Bias Calculation

The bias term \(b\) was computed using the average over all support vectors:

$$
b =
\frac{1}{N_{sv}}
\sum_{s \in SV}
\left[
t_s -
\sum_{m \in SV}
\alpha_m t_m k(x_m,x_s)
\right]
$$

---

## Kernels Implemented

Three kernels were implemented.

### RBF Kernel

$$
k(x,x') = \exp(-\gamma ||x-x'||^2)
$$

### Polynomial Kernel

$$
k(x,x') = (\gamma x^T x' + r)^d
$$

### Sigmoid Kernel

$$
k(x,x') = \tanh(\gamma x^T x' + r)
$$

All experiments used the **RBF kernel** with

$$
\gamma = 10^{-4}
$$

because it produced the most stable results.

---

# Task 2 — Prediction

For a new input \(x\), predictions are computed using

$$
y(x) =
\mathrm{sign}
\left(
\sum_{i \in SV}
\alpha_i t_i k(x_i,x) + b
\right)
$$

Only support vectors contribute to the prediction.

---

## Prediction Pipeline

1. Compute kernel matrix between **test samples and support vectors**

$$
K_{test,sv} \in \mathbb{R}^{n_{test} \times N_{sv}}
$$

2. Multiply by the weighted coefficients

$$
(\alpha_{sv} \odot t_{sv})
$$

3. Add bias \(b\)

4. Apply `sign()` to obtain the predicted label.

---

## Computational Complexity

Prediction complexity is

$$
O(N_{sv})
$$

Therefore, the number of support vectors directly impacts inference time.

---

# Task 3 — Multiclass Classification

Because SVM is a **binary classifier**, two strategies were implemented for the 10-class dataset.

---

## One-vs-Rest (OvR)

- Train **10 classifiers**
- Each classifier separates **one class vs all others**
- Prediction selects the class with the **highest decision score**

---

## One-vs-One (OvO)

- Train classifiers for **every pair of classes**

Total classifiers:

$$
\binom{10}{2} = 45
$$

Each classifier is trained using only the two relevant classes.

Prediction is determined by **majority voting** across all classifiers.

---

# Results

RBF kernel with

$$
C = 1.0
$$

was used for the comparison.

| Metric | One-vs-Rest | One-vs-One |
|------|------|------|
| Classifiers trained | 10 | 45 |
| Training time | 87.4 s | 7.3 s |
| Prediction time | 0.07 s | 0.24 s |
| Overall accuracy | 66.0% | 73.0% |

---

## Per-Class Accuracy

| Class | OvR | OvO |
|------|------|------|
| T-shirt/top | 58.82% | 58.82% |
| Trouser | 91.67% | 91.67% |
| Pullover | 0.00% | 65.22% |
| Dress | 76.19% | 80.95% |
| Coat | 100.00% | 66.67% |
| Sandal | 88.89% | 100.00% |
| Shirt | 0.00% | 31.58% |
| Sneaker | 76.47% | 73.53% |
| Bag | 93.75% | 81.25% |
| Ankle boot | 90.91% | 86.36% |

---

## Discussion

The **One-vs-One strategy improved accuracy by 7 percentage points**.

The One-vs-Rest model failed on **Pullover** and **Shirt**, achieving **0% accuracy**. This occurs because the negative class contains a heterogeneous mixture of nine categories.

In contrast, OvO reduces each problem to **pairwise classification**, which produces simpler decision boundaries.

Although OvO trains faster due to smaller optimization problems, prediction requires evaluating **45 classifiers**, which increases inference time.

---

# Task 4 — Hyperparameter Tuning (Regularization Parameter C)

The parameter \(C\) controls the tradeoff between **margin width and classification error**.

- Large \(C\): narrow margin, lower tolerance for errors  
- Small \(C\): wider margin, stronger regularization  

---

## Cross-Validation Strategy

- **3-fold stratified cross-validation**
- **400-sample balanced subset**
- **OvR classifier**

---

## Stage 1 — Coarse Search

Search range:

$$
C \in [0.1, 10^4]
$$

```python
np.logspace(-1, 4, 12)
