# ML Lab 401: Support Vector Machines — Lab Report

---

## Task 1: SVM Binary Classifier with Non-Linear Kernels

The binary SVM was implemented by solving the dual optimization problem from Bishop (Equations 7.32–7.34). The problem was rewritten in a form compatible with `cvxopt.solvers.qp`.

Let

$$
\mathbf{T} = \mathrm{diag}(\mathbf{t})
$$

be the diagonal label matrix and $K$ the kernel matrix.

The dual objective is:

$$
\min_{\boldsymbol{\alpha}}
\frac{1}{2}\boldsymbol{\alpha}^\top (\mathbf{T} K \mathbf{T}) \boldsymbol{\alpha} - \mathbf{1}^\top \boldsymbol{\alpha}
$$

subject to:

### Box constraints

$$
0 \le \alpha_i \le C
$$

encoded as:

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
\mathbf{0}
\end{bmatrix}
$$

### Equality constraint

$$
\mathbf{t}^\top \boldsymbol{\alpha} = 0
$$

---

### Support Vectors

Support vectors were identified as training points where

$$
\alpha_i > 10^{-5}
$$

The bias $b$ was computed using:

$$
b = \frac{1}{N_{sv}}
\sum_{s \in SV}
\left[
t_s -
\sum_{m \in SV}
\alpha_m t_m k(\mathbf{x}_m, \mathbf{x}_s)
\right]
$$

---

### Kernels Implemented

**RBF kernel**

$$
k(\mathbf{x}, \mathbf{x}') =
\exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)
$$

**Polynomial kernel**

$$
k(\mathbf{x}, \mathbf{x}') =
(\gamma \mathbf{x}^\top \mathbf{x}' + r)^d
$$

**Sigmoid kernel**

$$
k(\mathbf{x}, \mathbf{x}') =
\tanh(\gamma \mathbf{x}^\top \mathbf{x}' + r)
$$

The RBF kernel with

$$
\gamma = 10^{-4}
$$

was used for all experiments.

---

## Task 2: Prediction

For a new input $\mathbf{x}$:

$$
y(\mathbf{x}) =
\mathrm{sign}
\left(
\sum_{i \in SV}
\alpha_i t_i k(\mathbf{x}_i, \mathbf{x})
+ b
\right)
$$

Implementation steps:

- Compute test–support kernel matrix

$$
K_{\text{test,sv}} \in \mathbb{R}^{n_{\text{test}} \times N_{sv}}
$$

- Multiply by $(\boldsymbol{\alpha}_{sv} \odot \mathbf{t}_{sv})$
- Add $b$
- Apply `sign()`

Inference cost:

$$
\mathcal{O}(N_{sv})
$$

---

## Task 3: Multiclass Classification — One-vs-Rest vs One-vs-One

### One-vs-Rest (OvR)

- Train $K = 10$ binary classifiers
- Each classifier separates one class vs all others
- Choose class with highest decision score

### One-vs-One (OvO)

- Train $\binom{10}{2} = 45$ binary classifiers
- Each classifier trained on only two classes
- Prediction by majority vote

---

### Results (RBF kernel, $C = 1.0$)

| Metric | One-vs-Rest | One-vs-One |
|--------|------------|------------|
| Classifiers trained | 10 | 45 |
| Training time | 87.4 s | 7.3 s |
| Prediction time | 0.07 s | 0.24 s |
| Overall accuracy | 66.00% | 73.00% |

---

### Per-Class Accuracy

| Class | OvR | OvO |
|--------|------|------|
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

### Analysis

- OvO improves accuracy by 7 percentage points.
- OvR fails on Pullover and Shirt (0%).
- OvR struggles because the "rest" class is heterogeneous.
- OvO simplifies each boundary to pairwise separation.
- OvO trains faster because each QP is smaller.
- OvO predicts slower due to voting across 45 models.

---

## Task 4: Hyperparameter Tuning — Regularization Parameter $C$

$C$ controls the margin–slack tradeoff:

- Large $C$: narrow margin, high variance
- Small $C$: wider margin, more regularization

### Strategy

- 3-fold stratified cross-validation
- 400-sample balanced subset
- OvR used for efficiency

---

### Stage 1 — Coarse Search

$$
C \in [0.1, 10^{4}]
$$

```python
np.logspace(-1, 4, 12)
```

---

### Stage 2 — Fine Search

$$
C \in
\left[
10^{\log_{10}(C^*) - 0.5},
10^{\log_{10}(C^*) + 0.5}
\right]
$$

```python
np.logspace(np.log10(C_star) - 0.5,
            np.log10(C_star) + 0.5,
            8)
```

---

### Results

| Stage | Best C | CV Accuracy |
|--------|--------|------------|
| Coarse | 433 | 79.00% ± 0.98% |
| Fine | 367.2 | 79.50% ± 0.87% |

Final choice:

$$
C = 367.2
$$

---

## Task 5: Confusion Matrix Observations

Final model:

- OvR
- $C = 367.2$
- RBF kernel
- $\gamma = 10^{-4}$
- Trained on 1800 samples
- Tested on 200 samples

### Observations

Well-separated classes:
- Trouser
- Sandal
- Sneaker
- Bag
- Ankle boot

Confusable classes:
- T-shirt/top
- Shirt
- Pullover
- Coat

Errors are concentrated among visually similar garments.

---

## Summary

| Task | Key Finding |
|------|------------|
| SVM Implementation | Dual QP via `cvxopt`; support vectors at $\alpha_i > 10^{-5}$ |
| Prediction | $\mathcal{O}(N_{sv})$ inference |
| OvR vs OvO | OvO +7% accuracy |
| Hyperparameter Tuning | Optimal $C \approx 367$ |
| Confusion Analysis | Errors among similar top-wear categories |

---

**Dataset:** Fashion-MNIST (2,000 samples)  
**Kernel:** RBF ($\gamma = 10^{-4}$)  
**Final $C$:** 367.2  
**Implementation:** NumPy + cvxopt
