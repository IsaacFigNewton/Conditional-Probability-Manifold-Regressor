from sklearn.metrics import *
from sklearn.model_selection import KFold

# Prepare for k-fold cross-validation
debugging = 0
default_n_neighbors = 1
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
colormap = {
    0: 'red',
    1: 'green'
}
# Define the scoring metrics
scoring = {
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score)
}