## Overview
This project implements and compares **7 different decision tree variants** using the Breast Cancer Wisconsin dataset from Scikit-Learn.

The objective is to analyze how different tree configurations affect classification accuracy.

---

## Algorithms Implemented

1. CART using Gini Index  
2. ID3 using Entropy  
3. Decision Tree with Max Depth Constraint  
4. Decision Tree with Minimum Samples Split  
5. Decision Tree with Minimum Samples Leaf  
6. Pruned Decision Tree (Cost Complexity Pruning)  
7. Extra Trees Classifier

---

## Dataset Used
Breast Cancer Wisconsin Dataset  
- Instances: 569  
- Features: 30  
- Classes:
  - Malignant
  - Benign

Dataset loaded using:

```python
from sklearn.datasets import load_breast_cancer
```

---

## Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

