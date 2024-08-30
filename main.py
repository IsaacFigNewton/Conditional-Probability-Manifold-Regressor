import pandas as pd
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, \
    StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, \
    MultinomialNB
from sklearn.svm import SVC as SVM
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, \
    recall_score, \
    accuracy_score, \
    f1_score, \
    make_scorer
from sklearn.model_selection import GridSearchCV

from config import *
from modules.PMFRegressor import PMFRegressor
from modules.util import *


class_map = dict()
# Lists to store evaluation metrics
precisions, recalls, accuracies, f1_scores = [], [], [], []
avg_scores = dict()

def cv_dt(X, y, model_name):
    # Create a pipeline with preprocessor and DecisionTreeClassifier
    pipeline = Pipeline([
        (model_name, DecisionTreeClassifier())
    ])

    avg_scores[model_name] = cv_results_to_dict(pipeline, X, y)

    # Fit the pipeline to the data
    pipeline.fit(X, y)

    # # Create a figure and axis with a wider x-axis
    # fig, ax = plt.subplots(figsize=(30, 5))
    #
    # # Plot the decision tree
    # sk.tree.plot_tree(pipeline.named_steps['decision tree classifier'],
    #                   max_depth=5,
    #                   feature_names=transformed_feature_names,
    #                   fontsize=5,
    #                   ax=ax)

    plt.show()


def cv_knn(X, y, model_name):
    # Create a pipeline with preprocessor and KNeighborsClassifier
    pipeline = Pipeline([
        (model_name, KNeighborsClassifier(n_neighbors=default_n_neighbors))
    ])

    avg_scores[model_name] = cv_results_to_dict(pipeline, X, y)

def cv_nb(X, y, model_name):
    # Create a pipeline with preprocessor and GaussianNB classifier
    pipeline = Pipeline([
        (model_name, GaussianNB())
    ])

    avg_scores[model_name] = cv_results_to_dict(pipeline, X, y)

def cv_svm(X, y, model_name):
    # Create a pipeline with preprocessor and SVM classifier
    pipeline = Pipeline([
        (model_name, SVM())
    ])

    avg_scores[model_name] = cv_results_to_dict(pipeline, X, y)

def cv_mlp(X, y, model_name):
    # Create a pipeline with preprocessor and MLP classifier
    pipeline = Pipeline([
        (model_name, MLP())
    ])

    avg_scores[model_name] = cv_results_to_dict(pipeline, X, y)

def cv_rf(X, y, model_name):
    # Create a pipeline with preprocessor and RandomForestClassifier classifier
    pipeline = Pipeline([
        (model_name, RandomForestClassifier())
    ])

    avg_scores[model_name] = cv_results_to_dict(pipeline, X, y)

def cv_other_models(X, y):
    cv_dt(X, y, 'decision tree classifier')
    cv_knn(X, y, 'knn classifier')
    cv_nb(X, y, 'gaussian naive bayes classifier')
    cv_svm(X, y, 'svm classifier')
    cv_mlp(X, y, 'mlp classifier')
    cv_rf(X, y, 'rf classifier')

def cv_pmf(X, y):
    # Define the parameter grid
    param_grid = {
        'regressor__n_neighbors': range(1, 3),
        'regressor__max_bins': range(10, 20, 10)
    }

    # Create a scorer dictionary
    scorers = {
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }

    pipeline = Pipeline([
        ("regressor", PMFRegressor())
    ])

    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=kf,  # Use the same KFold object as before
        scoring=scorers,
        refit='f1',  # Refit using the best F1 score
        return_train_score=False,
        n_jobs=-1  # Use all available cores
    )

    # Fit the GridSearchCV object
    grid_search.fit(X, y.map(class_map))

    # Get the results
    results = pd.DataFrame(grid_search.cv_results_)

    # Process the results
    avg_scores["custom"] = dict()
    f1_by_k_bin_count = {
        "k": list(),
        "bin_count": list(),
        "f1": list()
    }

    for i, row in results.iterrows():
        k = row['param_regressor__n_neighbors']
        bin_count = row['param_regressor__max_bins']
        if k not in avg_scores["custom"]:
            avg_scores["custom"][k] = dict()
        avg_scores["custom"][k][bin_count] = {
            "precision": row['mean_test_precision'],
            "recall": row['mean_test_recall'],
            "accuracy": row['mean_test_accuracy'],
            "f1": row['mean_test_f1'],
            "std_precision": row['std_test_precision'],
            "std_recall": row['std_test_recall'],
            "std_accuracy": row['std_test_accuracy'],
            "std_f1": row['std_test_f1']
        }
        f1_by_k_bin_count["k"].append(k)
        f1_by_k_bin_count["bin_count"].append(bin_count)
        f1_by_k_bin_count["f1"].append(row['mean_test_f1'])

    # Convert f1_by_k to DataFrame
    f1_by_k_bins = pd.DataFrame.from_dict(f1_by_k_bin_count)


    return grid_search, f1_by_k_bins

if __name__ == "__main__":
    # Load the dataset  ************************************************************************************************
    # Titanic
    dataset = pd.read_csv("https://raw.githubusercontent.com/dlsun/pods/master/data/titanic.csv")
    dataset.drop(columns=["name", "ticketno"], inplace=True)
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    # take a sample for faster testing
    dataset = dataset.loc[:200]
    print(len(dataset))

    # Identify categorical, continuous, and binary columns
    quantitative_vars = ["age", "fare"]
    categorical_vars = ["gender", "embarked", "class", "country"]
    X = quantitative_vars + categorical_vars

    y = ['survived']
    class_map = {key: i for i, key in enumerate(dataset[y[0]].unique())}

    print(dataset.head())

    # Preprocessing ****************************************************************************************************
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_vars),
            ('continuous', StandardScaler(), quantitative_vars)
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('to_dense', dense_transformer)
    ])

    pipeline.fit(dataset[X])
    transformed_feature_names = pipeline[:-1].get_feature_names_out()
    # transform the dataset
    transformed_dataset = pd.DataFrame(pipeline.transform(dataset[X]),
                                       columns=transformed_feature_names)
    transformed_X = transformed_dataset[transformed_feature_names]

    # add the y values back in, after resetting their indices
    transformed_dataset[y[0]] = dataset[y[0]].reset_index(drop=True)
    transformed_y = transformed_dataset[y[0]]
    print(transformed_dataset.head())

    # Validation *******************************************************************************************************
    print(f"Empty Rows:\n {transformed_dataset[transformed_dataset.isnull().any(axis=1)]}")
    print(f"Dataset size before dropping np.nan transformed entries: {len(transformed_dataset)}")
    transformed_dataset.dropna(inplace=True)
    print(f"Dataset size after dropping np.nan transformed entries: {len(transformed_dataset)}")

    # # Data Exploration *************************************************************************************************
    # dataset.plot.scatter(x=[quantitative_vars[0]], y=y)
    # plt.title(f"{quantitative_vars[0]} vs {y[0]}")
    # plt.show()

    # Cross-Validation  ************************************************************************************************
    # Cross-validate other models
    cv_other_models(transformed_X, transformed_y)
    # Cross-validate the PMF classifier
    grid_search, f1_by_k_bins = cv_pmf(transformed_X, transformed_y)

    # Get the best parameters and scores
    optimal_k = grid_search.best_params_['regressor__n_neighbors']
    optimal_bin_count = grid_search.best_params_['regressor__max_bins']
    optimal_run = avg_scores["custom"][optimal_k][optimal_bin_count]
    best_model = grid_search.best_estimator_

    # Get model predictions
    model = best_model.fit(transformed_X, transformed_y.map(class_map))
    y_pred = pd.Series(model.predict(transformed_X))

    # Model Stats Analysis  ********************************************************************************************
    print(f"Optimal k: {optimal_k}")
    print(f"Optimal max bin count: {optimal_bin_count}")
    print(f"Best F1 score: {optimal_run['f1']}")
    print(f1_by_k_bins.head())
    for col in f1_by_k_bins.columns:
        if col != "f1":
            plt.plot(f1_by_k_bins[col], f1_by_k_bins["f1"], marker='o')
            plt.title(f"F1 Score vs. {col}")
            plt.xlabel(col)
            plt.ylabel("F1 Score")
            plt.show()
    print()
    print("Average Scores for Best Run (optimal k and bin_count):")
    print(f"n_neighbors = {optimal_k}")
    print(f"Precision: {optimal_run['precision']:.4f} (+/- {optimal_run['std_precision']:.4f})")
    print(f"Recall: {optimal_run['recall']:.4f} (+/- {optimal_run['std_recall']:.4f})")
    print(f"Accuracy: {optimal_run['accuracy']:.4f} (+/- {optimal_run['std_accuracy']:.4f})")
    print(f"F1 Score: {optimal_run['f1']:.4f} (+/- {optimal_run['std_f1']:.4f})")
    print()
    print()
    transformed_quant_names = ["continuous__" + quantitative_vars[i] for i in range(len(quantitative_vars))]
    plot_types_colors = [("True Class", transformed_y.map(colormap)),
                         ("Predicted Class", y_pred.map(colormap))]

    if len(quantitative_vars) > 1:
        for plot_type, color in plot_types_colors:
            plt.scatter(transformed_dataset[transformed_quant_names[0]], \
                        transformed_dataset[transformed_quant_names[1]], \
                        c=color, \
                        alpha=0.5)

            # Add labels and title
            plt.title(plot_type)
            plt.xlabel(transformed_quant_names[0])
            plt.ylabel(transformed_quant_names[1])
            plt.yscale('log')
            plt.show()

    # Compare Models  ********************************************************************************************
    for model_name, scores in avg_scores.items():
        if model_name != "custom":
            print(f"{model_name}:")
            print(f"Precision: {scores['precision']:.4f} (+/- {scores['std_precision']:.4f})")
            print(f"Recall: {scores['recall']:.4f} (+/- {scores['std_recall']:.4f})")
            print(f"Accuracy: {scores['accuracy']:.4f} (+/- {scores['std_accuracy']:.4f})")
            print(f"F1 Score: {scores['f1']:.4f} (+/- {scores['std_f1']:.4f})")
            print()
        else:
            print(f"{model_name}:")
            print(f"Optimal n_neighbors = {optimal_k}")
            print(f"Precision: {optimal_run['precision']:.4f} (+/- {optimal_run['std_precision']:.4f})")
            print(f"Recall: {optimal_run['recall']:.4f} (+/- {optimal_run['std_recall']:.4f})")
            print(f"Accuracy: {optimal_run['accuracy']:.4f} (+/- {optimal_run['std_accuracy']:.4f})")
            print(f"F1 Score: {optimal_run['f1']:.4f} (+/- {optimal_run['std_f1']:.4f})")
            print()

    # Prepare data for plotting
    models = list(avg_scores.keys())
    metrics = ['precision', 'recall', 'accuracy', 'f1']
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        values = []
        for model in models:
            if model == 'custom':
                values.append(optimal_run[metric])
            else:
                values.append(avg_scores[model][metric])
        ax.bar(x + i * width, values, width, label=metric)

    # Customize the plot
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0.3, 1)

    # Add value labels on top of each bar
    for i, metric in enumerate(metrics):
        values = []
        for model in models:
            if model == 'custom':
                values.append(optimal_run[metric])
            else:
                values.append(avg_scores[model][metric])
        for j, v in enumerate(values):
            ax.text(j + i * width, v, f'{v:.2f}', ha='center', va='bottom', rotation=0, fontsize=8)

    plt.tight_layout()
    plt.show()