from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf
import numpy as np

class Bin:
    def __init__(self, all_X, X, y, classes):
        self.contents = dict()
        self.X = X
        self.y = y
        self.classes_ = classes

        feature_count = self.X.shape[1]
        #   class likelihoods
        self.class_likelihoods = np.zeros(len(self.classes_))
        #   class likelihood variances
        self.class_likelihood_variances = np.zeros(len(self.classes_))
        #   bin width in each feature dimension
        self.widths = np.zeros(feature_count)
        #   bin center in each feature dimension
        self.centers = np.zeros(feature_count)
        #   mean binned point value in every feature dimension
        self.means = np.zeros(feature_count)

        # Covariance matrix of binned points using LedoitWolf for shrinkage and stability
        lw = LedoitWolf().fit(self.X)
        self.covariance_matrix = lw.covariance_

        for j in range(feature_count):
            # get the values of all the points in a certain feature dimension
            feature_values = self.X[:, j]
            self.widths[j] = np.max(feature_values) - np.min(feature_values)
            self.centers[j] = np.mean([np.max(feature_values), np.min(feature_values)])
            self.means[j] = np.mean(feature_values)

            # get the indices of all points in the dataset that lie within the bin's bounds
            min_bin_value = self.centers[j] - self.widths[j] / 2
            max_bin_value = self.centers[j] + self.widths[j] / 2
            self.get_contents(j, all_X, min_bin_value, max_bin_value)
        
    def get_contents(self, feature_idx, all_X, min_bin_value, max_bin_value):
        # filter all_x down to the entries such that min_bin_value <= all_X[:, feature_idx] <= max_bin_value
        mask = (all_X[:, feature_idx] >= min_bin_value) & (all_X[:, feature_idx] <= max_bin_value)

        # Apply the mask to filter rows and extract just the entries' indices
        self.contents[feature_idx] = np.where(mask)[0]


#TODO: IMPLEMENT THIS CORRECTLY
# bins indexed as [sample_size, data_point_index, likelihood (0) or likelihood_variance (1), class]
def merge_bin_stats(bins, classes):
    combined_class_likelihoods = np.zeros(classes)
    combined_class_likelihood_variances = np.zeros(classes)
    # Combining covariance matrices and means using a Kalman-like filter
    for bin_obj in bins:
        combined_class_likelihoods += bin_obj.class_likelihoods
        combined_class_likelihood_variances += bin_obj.class_likelihood_variances

    combined_class_likelihoods /= len(bins)
    combined_class_likelihood_variances /= len(bins)

    return combined_class_likelihoods, combined_class_likelihood_variances