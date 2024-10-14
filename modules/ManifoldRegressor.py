import numpy as np
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, \
    check_is_fitted, \
    check_array
from sklearn.preprocessing import FunctionTransformer, \
    KBinsDiscretizer
import pandas as pd
from sklearn.neighbors import KDTree
import tensorflow as tf

from config import *
from Bin import *

class ManifoldRegressor(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors_train=0, n_neighbors_test=1, alpha=1e-9):
        # Max bin size for training discretization
        self.n_neighbors_train = n_neighbors_train
        # Max bin size for testing discretization
        self.n_neighbors_test = n_neighbors_test
        # minimum weighted probability (0 implies that the feature of datapoints belonging to a class never takes on
        # a given value)
        self.alpha = alpha

        #  DO NOT APPEND UNDERSCORES TO X AND y
        self.X = None
        self.y = None
        self.classes_ = None
        self.priors = None
        # Store X as a kd tree for easier searching during testing
        self.kd_tree = None

        # Store the class conditional probabilities/likelihood binning information for all training points
        self.class_likelihoods = None
        # ... and their associated variances
        self.class_likelihood_variances = None


    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X = X
        self.y = y
        self.classes_ = np.unique(y)

        self.kd_tree = KDTree(X)
        self.priors = {c: np.sum(y == c) / X.shape[0] for c in self.classes_}
        self.class_likelihoods, self.class_likelihood_variances = self.get_aggregated_bin_stats(X=self.X,
                                                                n_neighbors=self.n_neighbors_train)
        # Return the classifier
        return self


    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        y = np.empty(X.shape[0], dtype=self.classes_.dtype)

        aggregated_class_likelihoods, aggregated_class_likelihood_variances = self.get_aggregated_bin_stats(X=X,
                                                             n_neighbors=self.n_neighbors_test)

        for i in range(X.shape[0]):
            predicted_class = max(aggregated_class_likelihoods, key=aggregated_class_likelihoods.get)
            y[i] = predicted_class

        return y


    def get_aggregated_bin_stats(self, X, n_neighbors):
        # Initialize ClassBinCondProbs as a tensor for conditional probabilities
        ClassBinCondProbs = tf.Variable(tf.zeros((n_neighbors, X.shape[0], len(self.classes_)), dtype=tf.float32))
        aggregated_class_likelihoods = []
        aggregated_class_likelihood_variances = []

        # for each data point
        for i in range(X.shape[0]):
            # Find the z nearest neighbors using the KDTree
            distances, indices = self.kd_tree.query(X[i], k=n_neighbors)

            bins_list = []
            # for each bin size
            for sample_size in range(n_neighbors):
                neighbor_indices = indices[0][:sample_size + 1]
                # Create a new Bin object
                bin_obj = Bin(self.X[neighbor_indices],
                              self.y[neighbor_indices],
                              self.classes_)
                # get the properties associated with the current bin, which include:
                #   class likelihoods
                #   class likelihood variances
                #   mean binned point value in every feature dimension
                #   covariance matrix of binned points
                #   bin width in each feature dimension
                #   bin center in each feature dimension
                bin_obj.calculate_bin_properties()
                bins_list.append(bin_obj)

                # Calculate class conditional probabilities for each class
                class_cond_probs = bin_obj.calculate_class_cond_probs()
                ClassBinCondProbs[sample_size, i].assign(class_cond_probs)

            # Combine statistics using Kalman filter and store
            combined_class_probs, combined_class_prob_variances = Bin.kalman_filter_combination(ClassBinCondProbs)
            # the aggregated likelihoods of each class
            aggregated_class_likelihoods.append(combined_class_probs)
            # the aggregated covariance matrices of each class
            aggregated_class_likelihood_variances.append(combined_class_prob_variances)

        return aggregated_class_likelihoods, aggregated_class_likelihood_variances