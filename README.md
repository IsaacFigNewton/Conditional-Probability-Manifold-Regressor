# Probability Mass Function Regressor v1  

## Goals  
- Estimate the class probability distribution of a dataset using KNN at various granularities (similar to a Kernel Density Estimator)
- Generalize nonlinear relationships better than standard decision trees or Gaussian Naïve Bayes while retaining a similar level of interpretability.  

## Definitions  
**Let:**  
- The training datapoint count be represented as $n$.  
- The feature count be represented as $m$.  
- The class count be represented as $l$.  
- The training dataset be denoted as:  
  $$A = \{ a_0, a_1, a_2, \dots, a_i, \dots, a_n \}$$
- The dimensions of the feature space in which all $a$ in $A$ reside be denoted as:  
  $$D = \{ f_0, f_1, f_2, \dots, f_j, \dots, f_m \}$$
- The set of classes be denoted as:  
  $$C = \{ c_0, c_1, c_2, \dots, c_k, \dots, c_l \}$$
- A testing data point be denoted as $x$.

## Steps  

### Fitting  
- Calculate local PMF values for each $a_i$ in $A$ (to avoid redundant computation later) by finding:  
  $$P(\text{class} \mid \text{value of } a_i f_1 \land \text{value of } a_i f_2 \land \dots)$$
  using Bayes’ Theorem.  
- In other words, compute the probability of $a_i$’s value in $f_j$, denoted as $p_{ij}$.  
- Find  
  $$i, j = 0 \dots n, m \quad \Rightarrow \quad p_{ij}$$
- Multiply by $P(\text{class})$.  
- Repeat this process for all classes.  
- Essentially, apply Bayes’ Theorem to all training data points for each class.  

### Use KNN with $k$ nearest neighbors in each dimension  
This ensures a standard "resolution" of the algorithm in every dimension but provides:  
- A maximum of $k \times m$ neighbors to consider for estimating the value of $x$.  
- A minimum of $k$ neighbors for consideration.  

### Compute Weighted Average of All Neighbors' Likelihoods  

## For the Future  
- Recursively perform this process on the $l$th neighbors-of-neighbors of the original, unknown seed datapoint.  
- Calculate the weighted mean of the probabilities using inverse-distance kernel weighting.  
- Consider additional weighting based on variability in class attributable to each feature.  
