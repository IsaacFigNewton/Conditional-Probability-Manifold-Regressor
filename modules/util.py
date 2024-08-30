import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_validate

from config import *


# Custom transformer to convert sparse matrix to dense
def to_dense(X):
    return np.array(X.todense()) if hasattr(X, 'todense') else X

dense_transformer = FunctionTransformer(to_dense)
[5]
def print_tree(tree, indent=0):
    # Iterate over the keys (features) in the tree
    for key, value in tree.items():
        print(' ' * indent + str(key))
        # If the value is a dictionary, recursively print the subtree
        if isinstance(value, dict):
            print_tree(value, indent + 4)
        else:
            print(' ' * (indent + 4) + str(value))
[6]
def cv_results_to_dict(pipeline, X, y):
    model_name = pipeline.steps[-1][0]

    # Perform cross-validation
    cv_results = cross_validate(pipeline,
                                X,
                                y,
                                cv=kf,
                                scoring=scoring,
                                return_train_score=False)

    # Calculate average scores and standard deviations
    avg_scores = {
            "precision": cv_results['test_precision'].mean(),
            "recall": cv_results['test_recall'].mean(),
            "accuracy": cv_results['test_accuracy'].mean(),
            "f1": cv_results['test_f1'].mean(),
            "std_precision": cv_results['test_precision'].std(),
            "std_recall": cv_results['test_recall'].std(),
            "std_accuracy": cv_results['test_accuracy'].std(),
            "std_f1": cv_results['test_f1'].std(),
    }

    if debugging > 0:
        # Print average scores
        print("\nAverage Scores:")
        print(f"Precision: {avg_scores['precision']:.4f} (+/- {avg_scores['std_precision']:.4f})")
        print(f"Recall: {avg_scores['recall']:.4f} (+/- {avg_scores['std_recall']:.4f})")
        print(f"Accuracy: {avg_scores['accuracy']:.4f} (+/- {avg_scores['std_accuracy']:.4f})")
        print(f"F1 Score: {avg_scores['f1']:.4f} (+/- {avg_scores['std_f1']:.4f})")

        # If you want to see individual fold scores:
        for fold, (precision, recall, accuracy, f1) in enumerate(zip(
            cv_results['test_precision'],
            cv_results['test_recall'],
            cv_results['test_accuracy'],
            cv_results['test_f1']
        ), 1):
            print(f"\nFold {fold}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")

    return avg_scores