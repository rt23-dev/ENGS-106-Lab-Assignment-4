# ML Lab 401 — Support Vector Machines

**Name:** Rohan Taneja  
**Course:** ML Lab 401  
**Dataset:** Fashion-MNIST  
**Implementation:** NumPy + CVXOPT  

---

# 1. Introduction

In this lab we implemented a **Support Vector Machine (SVM)** classifier from first principles using the dual optimization formulation described in Bishop's *Pattern Recognition and Machine Learning*.  

The objectives of the lab were:

- Implement a binary SVM using quadratic programming
- Implement kernel functions
- Extend the classifier to multiclass classification
- Compare **One-vs-Rest (OvR)** and **One-vs-One (OvO)**
- Perform hyperparameter tuning for the regularization parameter \(C\)
- Analyze model performance using confusion matrices

The experiments were conducted on a subset of the **Fashion-MNIST dataset**, which consists of grayscale images of clothing items.

---

# 2. Binary SVM Formulation

The SVM was implemented using the **dual optimization problem**.

Let the training labels be

$$
\mathbf{t} \in \{-1,1\}^N
$$

Define the diagonal label matrix:

$$
\mathbf{T} = \mathrm{diag}(\mathbf{t})
$$

and let \(K\) denote the kernel matrix.

The dual optimization problem is:

$$
\min_{\boldsymbol{\alpha}}\frac{1}{2}\boldsymbol{\alpha}^T (\mathbf{T}K\mathbf{T})\boldsymbol{\alpha} - \mathbf{1}^T \boldsymbol{\alpha}
$$

subject to

$$
0 \le \alpha_i \le C
$$

and

$$
\mathbf{t}^T \boldsymbol{\alpha} = 0
$$

The problem was solved using the **CVXOPT quadratic programming solver**.

---

# 3. Kernel Functions

Three kernel functions were implemented.

### Radial Basis Function (RBF)

$$
k(x,x') = \exp(-\gamma ||x-x'||^2)
$$

### Polynomial Kernel

$$
k(x,x') = (\gamma x^Tx' + r)^d
$$

### Sigmoid Kernel

$$
k(x,x') = \tanh(\gamma x^Tx' + r)
$$

After experimentation, the **RBF kernel** was selected for all experiments because it performed most reliably on the high-dimensional pixel data.

Parameters used:

\( \gamma = 10^{-4} \)

---

# 4. Support Vectors and Bias

Support vectors were defined as training points where

$$
\alpha_i > 10^{-5}
$$

The bias term was computed as

$$
b = \frac{1}{N_{sv}}
\sum_{s\in SV}
\left[
t_s -
\sum_{m\in SV} \alpha_m t_m k(x_m,x_s)
\right]
$$

---

# 5. Prediction

Predictions were computed using

$$
y(x) =
\text{sign}
\left(
\sum_{i\in SV}\alpha_i t_i k(x_i,x) + b
\right)
$$

Since only support vectors are retained after training, the prediction complexity becomes

$$
O(N_{sv})
$$

which is significantly smaller than using the full training dataset.

---

# 6. Multiclass Classification

Because Fashion-MNIST contains **10 classes**, two strategies were implemented.

---

## 6.1 One-vs-Rest (OvR)

- Train **10 binary classifiers**
- Each classifier distinguishes **one class vs all others**
- Prediction selects the class with the highest decision score

---

## 6.2 One-vs-One (OvO)

- Train **45 classifiers** (all class pairs)
- Each classifier is trained using only two classes
- Prediction uses **majority voting**

---

# 7. Dataset Examples

The following figure shows example images from the Fashion-MNIST dataset used in the experiment.

![Figure 1: Example Fashion-MNIST images](images/sample_images.png)

---

# 8. Kernel Behavior Visualization

The following figure illustrates the behavior of the kernel functions used in the model.

![Figure 2: Kernel function visualization](images/kernel_plot.png)

---

# 9. OvR vs OvO Results

Experiments were conducted using:

- RBF Kernel
- \(C = 1.0\)

| Metric | One-vs-Rest | One-vs-One |
|------|------|------|
| Classifiers trained | 10 | 45 |
| Training time | 87.4 s | 7.3 s |
| Prediction time | 0.07 s | 0.24 s |
| Accuracy | 66% | 73% |

### Discussion

The **One-vs-One strategy produced higher accuracy** because each classifier only needs to separate two classes, resulting in simpler decision boundaries.

However, OvO requires evaluating many classifiers during prediction.

---

# 10. Hyperparameter Tuning

The regularization parameter \(C\) controls the tradeoff between margin width and classification error.

- Large \(C\) → narrow margin, less regularization  
- Small \(C\) → wider margin, stronger regularization  

A **two-stage search** was used.

---

## Coarse Search

$$
C \in [0.1, 10^4]
$$

```python
np.logspace(-1,4,12)
```

---

## Fine Search

```python
np.logspace(np.log10(C_star)-0.5,
            np.log10(C_star)+0.5,
            8)
```

---

## Cross Validation Results

| Stage | Best C | CV Accuracy |
|------|------|------|
| Coarse | 433 | 79.0% |
| Fine | 367.2 | 79.5% |

Final value used:

$$
C = 367.2
$$

---

# 11. Cross-Validation Accuracy Plot

The following figure shows accuracy as a function of \(C\).

![Figure 3: Cross-validation accuracy vs C](images/cv_plot.png)

The results show that performance increases rapidly for small values of \(C\), then plateaus before slightly declining due to overfitting.

---

# 12. Confusion Matrix

The confusion matrix below shows the classification results of the final model.

![Figure 4: Confusion Matrix](images/confusion_matrix.png)

---

# 13. Error Analysis

The confusion matrix reveals several patterns:

### Well separated classes
- Trouser
- Sandal
- Sneaker
- Bag
- Ankle boot

These classes have visually distinctive shapes.

### Frequently confused classes
- T-shirt/top
- Shirt
- Pullover
- Coat

These categories share similar silhouettes in the raw pixel representation.

Without convolutional feature extraction, the classifier relies purely on pixel similarity.

---

# 14. Conclusion

In this lab we successfully implemented a Support Vector Machine from first principles.

Key findings:

- SVM training can be formulated as a **quadratic optimization problem**
- Kernel functions allow the model to learn **nonlinear decision boundaries**
- **One-vs-One classification outperformed One-vs-Rest**
- Hyperparameter tuning improved classification accuracy
- Errors mainly occur between visually similar clothing items

The experiment demonstrates both the strengths and limitations of SVMs when applied directly to raw pixel data.

---

# References

Christopher M. Bishop  
*Pattern Recognition and Machine Learning*  
Chapter 7 — Support Vector Machines
