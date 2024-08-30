import numpy as np
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, \
    check_is_fitted, \
    check_array
from sklearn.preprocessing import FunctionTransformer, \
    KBinsDiscretizer
from sklearn.neighbors import KernelDensity, \
    KNeighborsRegressor

from config import *

class PMFRegressor(ClassifierMixin, BaseEstimator):

    def __init__(self, max_bins=None, alpha=1e-9, n_neighbors=1):
        # Max bins for discretization, to limit memory usage
        self.max_bins = max_bins
        # minimum weighted probability (0 implies that the feature of datapoints belonging to a class never takes on
        # a given value)
        self.alpha = alpha
        # Number of neighbors to use in KNN estimation
        self.n_neighbors = n_neighbors

        #  DO NOT APPEND UNDERSCORES TO X AND y
        self.X = None
        self.y = None
        self.classes_ = None

        self.prior_class_probabilities = None
        self.pmf_dict = None
        self.knn_model_store = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X = X
        self.y = y
        self.classes_ = np.unique(y)

        # Estimate the probability mass function
        self.prior_class_probabilities = {class_value: self.get_prior_probability(class_value) for class_value in
                                          self.classes_}
        # create a multilevel dict to represent the weighted probabilities of every unique value of every feature for
        # every class
        self.pmf_dict = {class_value: {feature: {} for feature in range(self.X.shape[1])} for class_value in
                         self.classes_}
        # create a dict to store PMFs for each feature and unique value
        self.knn_model_store = {class_value: {feature: defaultdict() for feature in range(self.X.shape[1])} for
                                class_value in self.classes_}

        # Add the weighted probabilities of the unique value to the feature distribution set
        for class_value in self.classes_:
            # Calculating and storing PMFs
            for feature in range(self.X.shape[1]):
                # Get unique values for the current feature
                # default; discretize down to scale of datapoints
                unique_values = np.unique(self.X[:, feature])

                # alternate; discretize down to datapoints or max_bins
                if self.max_bins is not None and len(unique_values) > self.max_bins:
                    discretizer = KBinsDiscretizer(n_bins=self.max_bins, encode='ordinal', strategy='quantile')
                    unique_values = discretizer.fit_transform(unique_values.reshape(-1, 1)).flatten()

                # Calculate the weighted probability for each unique value
                for unique_value in unique_values:
                    # Calculate the probability distribution for the current feature and unique value
                    pmf_val, p_unique = self.calc_pmf_at_value(class_value, feature, unique_value)

                    self.pmf_dict[class_value][feature][unique_value] = pmf_val

                    # Create a KNN for the current feature
                    pmf = self.create_pmf_using_knn(class_value, feature)

                    # Store the KNN for the current feature and class
                    self.knn_model_store[class_value][feature] = pmf

        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        y = np.empty(X.shape[0], dtype=self.classes_.dtype)

        for i in range(X.shape[0]):
            datapoint = {j: X[i, j] for j in range(X.shape[1])}
            class_probabilities = self.calculate_class_probabilities(datapoint)

            predicted_class = max(class_probabilities, key=class_probabilities.get)
            y[i] = predicted_class

        return y

    # Main functions
    # *******************************************************************************************************************************************

    # Function to get prior probability of the class
    def get_prior_probability(self, class_value):
        return np.sum(self.y == class_value) / len(self.X)

    # Function to calculate probability distribution
    def calc_pmf_at_value(self, class_value, feature, unique_value):
        # Calculate P(unique_val | feature)
        mask = self.X[:, feature] == unique_value
        p_unique = np.sum(mask) / len(self.X)

        # Filter the dataset for the current class and feature value
        class_mask = self.y == class_value
        class_data_mask = mask & class_mask

        # Calculate P(unique_val | feature /\ class)
        p_unique_given_class = np.sum(class_data_mask) / np.sum(class_mask)

        # Avoid division by zero
        if p_unique > 0:
            # Calculate P(unique_val | feature /\ class)/P(unique_val | feature)
            pmf_value = p_unique_given_class / p_unique
        else:
            # No data for this unique value
            pmf_value = self.alpha

        return pmf_value, p_unique

    # Function to get values and their associated probabilities for the current feature and class
    def get_feature_class_vals_and_probs(self, class_value, feature):
        pmf_class_feature = self.pmf_dict[class_value][feature]

        # print(pmf_class_feature)

        # Get the values for the current feature and class
        values = np.array( \
            list( \
                pmf_class_feature.keys())).reshape(-1, 1)
        # Get the weighted probabilities for the current feature and class
        probabilities = np.array( \
            list( \
                pmf_class_feature.values()))

        return values, probabilities

    # Function to create a KNN for prob_distribution in the current feature and class
    def create_pmf_using_knn(self, class_value, feature):
        values, probabilities = self.get_feature_class_vals_and_probs(class_value, feature)

        # Create a kernel density estimator using Gaussian kernel
        knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights="distance").fit(values, probabilities)

        return knn

    def get_standardized_feature_contributions(self):
        # Get the sum of squares of all residuals

        # Get the residuals of all the points on the discretized PMF
        squared_residuals = self.pmf_df.sub(self.y, axis=0) ** 2

        # Get the sum of squares (SS) for each feature
        sum_of_squares = squared_residuals.sum()

        return np.divide(squared_residuals, sum_of_squares)

    # get the weights for each neighbor of a test datapoint in a given feature
    def get_knn_weighted_avg(self,
                             class_value,
                             feature, value,
                             neighbor_indices,
                             neighbor_distances):
        values, probabilities = self.get_feature_class_vals_and_probs(class_value, feature)
        neighbor_values = values[neighbor_indices]

        # handle edge cases where something went horribly wrong
        #   and where n_neighbors = 1 and thus the only weight will be 0
        if len(neighbor_values) == 0:
            return 0
        elif len(neighbor_values) == 1:
            return probabilities[neighbor_indices[0]]

        # get modified inverse distance weights
        standardized_distances = np.divide(neighbor_distances,
                                           np.sum(neighbor_distances))
        distance_weights = np.divide(1 - standardized_distances,
                                     np.sum(standardized_distances))

        weighted_avg = np.average(probabilities[neighbor_indices],
                                  weights=distance_weights)

        # print(neighbor_distances)
        # print(standardized_distances)
        # print(distance_weights)

        return weighted_avg

    # Function to calculate P(class | feature1_value /\ feature2_value /\ …)
    def calculate_class_probabilities(self, feature_values):
        if debugging > 1:
            print("calculate_class_probabilities() is broken for all k > 1")
        class_probabilities = {class_value: 1 for class_value in self.prior_class_probabilities.keys()}

        for class_value in self.prior_class_probabilities.keys():
            likelihood = 1
            for feature, value in feature_values.items():
                # feature_likelihood = self.knn_model_store[class_value][feature].predict(np.array([[value]]))
                # # Accumulate features' weighted probabilities
                # likelihood *= feature_likelihood

                datapoint = np.array([[value]])
                model = self.knn_model_store[class_value][feature]

                # Get the custom weighted KNN estimate for the current datapoint, feature, and class
                # Get the neighbors of the current datapoint and their indices in
                distances, indices = model.kneighbors(X=datapoint, n_neighbors=self.n_neighbors)

                # Get the weighted average of the neighbors' probabilities using built-in distance weighting
                datapoint_class_likelihood = model.predict(datapoint)

                # # Get the weighted average of the neighbors' probabilities using custom distance weighting
                # datapoint_class_likelihood = self.get_knn_weighted_avg(class_value,
                #                                                  feature,
                #                                                  datapoint,
                #                                                  indices,
                #                                                  distances)

                if debugging > 1:
                    print("Neighbor distances: ", distances)
                    print("Datapoint class likelihoods: ", datapoint_class_likelihood)

                # Accumulate features' weighted probabilities
                #   and multiply by the standardized feature contributions to the data's output
                likelihood *= datapoint_class_likelihood  # * self.standardized_feature_contributions[feature]

            # Get final class probability
            class_probability = likelihood * self.prior_class_probabilities[class_value]
            class_probabilities[class_value] = class_probability

        return class_probabilities

    # Helper functions
    # *******************************************************************************************************************************************

    # Map the features' weighted probability distributions to lists of features' values and their weighted probabilities
    def pmf_dict_to_dict(self):
        return {
            class_value: {
                feature: (list(self.pmf_dict[class_value][feature].keys()),
                          list(self.pmf_dict[class_value][feature].values()))
                for feature in range(self.X.shape[1])
            } for class_value in self.classes_
        }

    def print_weighted_prob_dist(self):
        print("Feature Distribution:")
        print(self.pmf_dict_to_dict())
