# HOST-CP

Authors: Soumi Das, Arshdeep Singh, Saptarshi Chatterjee, Suparna Bhattacharya, Sourangshu Bhattacharya

Paper: Finding High-Value Training Data Subset through Differentiable Convex Programming

This is the code repository for the paper "Finding High-Value Training Data Subset through Differentiable Convex Programming" accepted at ECML-PKDD 2021. We propose the method HOST-CP (High-value Online Subset selection of Training samples through differentiable Convex Programming) for selecting subsets in an online method.

# Pre-requisites

Python, NumPy, PyTorch, cvxpy, cvxpylayers, faiss, scikit-learn, Matplotlib

# Usage
This setup is provided for CIFAR10 dataset using ResNet-18 model. 

One needs to first store ResNet-18 checkpoint under resnet_checkp/ which is pre-trained on the entire dataset. ```python train.py``` can be used for this purpose.

Following this, running ```python subsetfind.py``` yields the subset based on the fraction one provides.

```python trainsub.py``` can be used to run training on the subset provided by the method.
