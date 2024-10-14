from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf
import numpy as np

class Bin:
    def __init__(self, X, y, classes):
        self.X = X
        self.y = y
        self.classes_ = classes

        #   class likelihoods
        self.class_likelihoods = None
        #   class likelihood variances
        self.class_likelihood_variances = None
        #   bin width in each feature dimension
        self.bin_widths = None
        #   bin center in each feature dimension
        self.bin_centers = None
        #   mean binned point value in every feature dimension
        self.bin_means = None
        #   covariance matrix of binned points
        self.bin_covariance_matrix = None

    # get the properties associated with the current bin
    # includes conditional probabilities for all classes, but, again, for only 1 bin
    # TODO: IMPLEMENT
    def calculate_bin_properties(self):
        feature_count = self.X.shape[1]
        # Calculate binWidths, binCenters, binMeans, and binCovarianceMatrix
        self.bin_widths = np.zeros(feature_count)
        self.bin_centers = np.zeros(feature_count)
        self.bin_means = np.zeros(feature_count)

        for j in range(feature_count):
            feature_values = self.bin_contents[:, j]
            self.bin_widths[j] = np.max(feature_values) - np.min(feature_values)
            self.bin_centers[j] = np.mean([np.max(feature_values), np.min(feature_values)])
            self.bin_means[j] = np.mean(feature_values)

        # Covariance matrix using LedoitWolf for shrinkage and stability
        lw = LedoitWolf().fit(self.bin_contents)
        self.bin_covariance_matrix = lw.covariance_

    def calculate_class_cond_probs(self):
        class_cond_probs = np.zeros(len(self.classes_))

        for class_idx in range(len(self.classes_)):
            prob_product = 1
            for j in range(self.bin_widths.shape[0]):
                bin_width = self.bin_widths[j] / 2
                bin_center = self.bin_centers[j]
                prob = multivariate_normal.pdf(bin_center, mean=self.bin_means[j],
                                               cov=self.bin_covariance_matrix[j, j])
                prob *= bin_width
                prob_product *= prob

            class_cond_probs[class_idx] = prob_product / self.priors[self.classes[class_idx]]

        return class_cond_probs

#TODO: IMPLEMENT
# bins_list of type tf.Variable(tf.zeros((n_neighbors, X.shape[0], len(self.classes_)), dtype=tf.float32))
def kalman_filter_combination(bins_list):
    # Combining covariance matrices and means using a Kalman-like filter

    return combined_class_likelihoods, combined_class_likelihood_variances