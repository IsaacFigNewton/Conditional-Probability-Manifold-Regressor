from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, \
    check_is_fitted, \
    check_array
from sklearn.neighbors import KDTree
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
        self.priors = {class_value: np.sum(self.y == class_value) / self.X.shape[0] for class_value in self.classes_}
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

        aggregated_class_likelihoods,\
        aggregated_class_likelihood_variances = self.get_aggregated_bin_stats(X=X,
                                                                            n_neighbors=self.n_neighbors_test)

        for i in range(X.shape[0]):
            predicted_class = max(aggregated_class_likelihoods, key=aggregated_class_likelihoods.get)
            y[i] = predicted_class

        return y


    def get_aggregated_bin_stats(self, X, n_neighbors):
        # Initialize class_bin_cond_probs as a tensor for conditional probabilities
        # indexed as [sample_size, data_point_index, likelihood (0) or likelihood_variance (1), class]
        class_bin_cond_probs = np.zeros((n_neighbors, X.shape[0], 2, len(self.classes_)))
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
                bin_obj = Bin(self.X,
                              self.X[neighbor_indices],
                              self.y[neighbor_indices],
                              self.classes_)
                # Get the class conditional probabilities based on the bin state
                bin_obj = self.calculate_class_cond_probs_for_bin(bin=bin_obj,
                                                                  method="bayes")
                bins_list.append(bin_obj)

                # Calculate class conditional probabilities for each class
                class_bin_cond_probs[sample_size, i, 0] = bin_obj.class_likelihoods

            # Combine statistics using Kalman filter and store
            #TODO: IMPLEMENT THIS
            combined_class_probs, combined_class_prob_variances = merge_bin_stats(class_bin_cond_probs,
                                                                                  self.classes_)
            # the aggregated likelihoods of each class
            aggregated_class_likelihoods.append(combined_class_probs)
            # the aggregated covariance matrices of each class
            aggregated_class_likelihood_variances.append(combined_class_prob_variances)

        return aggregated_class_likelihoods, aggregated_class_likelihood_variances


    def calculate_class_cond_probs_for_bin(self, bin, method="pdf"):
        # for each class
        for class_idx in range(len(self.classes_)):
            cond_prob = 1
            # for each feature
            for j in range(bin.X.shape[1]):
                if method == "pdf":
                    prob = multivariate_normal.pdf(bin.centers[j], mean=bin.means[j],
                                                   cov=bin.covariance_matrix[j, j])
                    # double-check that this is right
                    # use the bin width/2 as the max distance from the center to include points in the bin
                    prob *= bin.widths[j] / 2
                    cond_prob *= prob
                elif method == "bayes":
                    #TODO
                    # MODIFY THIS TO BE COMPATIBLE WITH CONTINUOUS PROBABILITY VALUES INSTEAD OF JUST 1 OR 0
                    mask = (self.y == self.classes_[class_idx]) & np.isin(np.arange(len(self.y)), bin.contents[j])
                    points_in_bin = bin.contents[j]
                    points_in_bin_and_class = self.X[mask]

                    # get the probability that a point in the bin is of a given class_idx
                    cond_prob *= points_in_bin_and_class.shape[0] / points_in_bin.shape[0]

            bin.class_likelihoods[class_idx] = cond_prob / self.priors[self.classes_[class_idx]]

            #TODO: CALCULATE class_likelihood_variances
            bin.class_likelihood_variances = bin.class_likelihoods[class_idx]/2

        return bin