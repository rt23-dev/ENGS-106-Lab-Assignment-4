# ML Lab 401 — Support Vector Machines

**Name:** Rohan Taneja  
**Dataset:** Fashion-MNIST  
**Implementation:** NumPy + CVXOPT  

---

# 1. Introduction

In this assignment we implement a **Support Vector Machine (SVM)** classifier from first principles using the dual optimization formulation described in Bishop’s *Pattern Recognition and Machine Learning*.  

The goals of this lab are to:

- Implement a binary SVM using quadratic programming
- Implement kernel functions
- Extend the classifier to multiclass classification
- Compare **One-vs-Rest (OvR)** and **One-vs-One (OvO)** strategies
- Perform hyperparameter tuning for the regularization parameter \(C\)
- Evaluate performance using confusion matrices

Experiments were conducted using a subset of the **Fashion-MNIST dataset**, which contains grayscale images of clothing items.

---

# 2. Dataset and Preprocessing

The dataset used is **Fashion-MNIST**, which contains \(28 \times 28\) grayscale images of clothing items across 10 categories.

For computational efficiency:

- Only **2000 samples** were used.
- The dataset was split using `train_test_split`.
- **90% training data**
- **10% test data**

Each image was represented as a **784-dimensional feature vector** obtained by flattening the pixel values.

### Example Image from Dataset

The following figure shows a randomly sampled image from the dataset.

<img width="790" height="832" alt="image" src="https://github.com/user-attachments/assets/d9002dc9-8f1c-4568-8ad5-7bb59a108fa4" />


---

# 3. Support Vector Machine Formulation

The SVM classifier was implemented using the **dual optimization formulation**.

Let the training labels be:

\[
t_i \in \{-1,1\}
\]

Define the diagonal label matrix:

\[
T = \text{diag}(t)
\]

Let \(K\) denote the kernel matrix.

The dual objective function is:

\[
\min_{\alpha}
\frac{1}{2}\alpha^T (TKT)\alpha - 1^T\alpha
\]

subject to

\[
0 \leq \alpha_i \leq C
\]

and

\[
t^T\alpha = 0
\]

The quadratic optimization problem was solved using the **CVXOPT solver**.

---

# 4. Kernel Functions

Three kernel functions were implemented.

### Radial Basis Function (RBF)

\[
k(x,x') = \exp(-\gamma ||x-x'||^2)
\]

### Polynomial Kernel

\[
k(x,x') = (\gamma x^Tx' + r)^d
\]

### Sigmoid Kernel

\[
k(x,x') = \tanh(\gamma x^Tx' + r)
\]

After testing, the **RBF kernel** was used for the main experiments since it provided the most stable performance on the high-dimensional image data.

---

# 5. Support Vectors and Bias

After solving the optimization problem, support vectors were identified as samples satisfying

\[
\alpha_i > 10^{-5}
\]

The bias term was computed using the support vectors according to the standard SVM formulation.

Only support vectors are retained after training, which significantly reduces prediction cost.

---

# 6. Prediction

Predictions for a new input \(x\) are computed as

\[
y(x) =
\text{sign}\left(
\sum_{i \in SV} \alpha_i t_i k(x_i,x) + b
\right)
\]

Since the model only uses support vectors, the prediction complexity scales with the number of support vectors rather than the full training set.

---

# 7. Multiclass Classification

Since Fashion-MNIST contains **10 classes**, two strategies were implemented.

---

## 7.1 One-vs-Rest (OvR)

- Train **10 binary classifiers**
- Each classifier separates **one class vs all other classes**
- The class with the highest decision score is selected

---

## 7.2 One-vs-One (OvO)

- Train classifiers for **every pair of classes**
- Total classifiers:

\[
\binom{10}{2} = 45
\]

- Prediction uses **majority voting** across all classifiers.

---

# 8. Model Performance

Both multiclass approaches were evaluated in terms of:

- training time
- prediction time
- classification accuracy

### Comparison of OvR and OvO

| Metric | One-vs-Rest | One-vs-One |
|------|------|------|
| Number of classifiers | 10 | 45 |
| Training time | Higher | Lower |
| Prediction time | Faster | Slower |
| Accuracy | Lower | Higher |

OvO generally achieved better accuracy because each classifier only separates two classes, resulting in simpler decision boundaries.

---

# 9. Hyperparameter Tuning

The regularization parameter \(C\) controls the tradeoff between margin width and classification error.

- Large \(C\) → narrow margin, lower bias
- Small \(C\) → wider margin, stronger regularization

A **two-stage search strategy** was used.

### Stage 1 — Coarse Search

Values of \(C\) were explored on a logarithmic scale:

```
np.logspace(-1,4,12)
```

### Stage 2 — Fine Search

A finer search was performed around the best coarse value.

The final selected value was approximately:

\[
C \approx 367
\]

---

# 10. Cross-Validation Results

The following figure shows validation accuracy as a function of \(C\).

<img width="1589" height="611" alt="image" src="https://github.com/user-attachments/assets/f4d08152-e27f-4508-aea4-ee6d76cfb565" />

The results show that performance increases rapidly for small values of \(C\), then stabilizes before slightly decreasing due to overfitting.

---

# 11. Confusion Matrix

The final classifier was evaluated using a confusion matrix.

<img width="498" height="438" alt="image" src="https://github.com/user-attachments/assets/d5ed7c3b-4d8b-4e15-9ff4-46ec1731f87f" />


---

# 12. Error Analysis

The confusion matrix reveals several patterns.

### Well-classified categories

- Trouser
- Sandal
- Sneaker
- Bag
- Ankle boot

These classes have distinct shapes and are easier to separate.

### Frequently confused categories

- T-shirt/top
- Shirt
- Pullover
- Coat

These items have similar silhouettes, making them harder to distinguish using raw pixel features.

---

# 13. Conclusion

In this assignment we implemented an SVM classifier from scratch and applied it to the Fashion-MNIST dataset.

Key findings:

- SVM training can be formulated as a **quadratic optimization problem**.
- Kernel functions enable the learning of **nonlinear decision boundaries**.
- **One-vs-One classification generally performs better** than One-vs-Rest.
- Hyperparameter tuning of \(C\) significantly affects model performance.
- Most classification errors occur between visually similar clothing categories.

This experiment demonstrates both the strengths and limitations of SVMs when applied directly to raw image pixels.

---

# References

Christopher M. Bishop  
*Pattern Recognition and Machine Learning*  
Chapter 7 — Support Vector Machines
