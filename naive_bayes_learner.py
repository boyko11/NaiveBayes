import numpy as np
import math


class NaiveBayesLearner:

    def __init__(self):
        self.labels_probability = None
        self.means = {}
        self.st_devs = {}

    def train(self, feature_data, labels):

        unique_labels, label_counts = np.unique(labels, return_counts=True)
        self.labels_probability = label_counts/labels.size

        for unique_label in unique_labels:
            record_indices_for_this_label = np.where(labels == unique_label)[0]
            self.means[unique_label] = np.mean(feature_data[record_indices_for_this_label, :], axis=0)
            self.st_devs[unique_label] = np.std(feature_data[record_indices_for_this_label, :], axis=0)

    def predict(self, feature_data):

        labels_probability_given_this_features = {}
        for label, label_probability in enumerate(self.labels_probability):
            features_probabilities_given_label = self.gaussian_probability(feature_data, self.means[label],
                                                                           self.st_devs[label])
            labels_probability_given_this_features[label] = np.prod(features_probabilities_given_label) * \
                                                            self.labels_probability[label]

        return max(labels_probability_given_this_features, key = labels_probability_given_this_features.get)

    @staticmethod
    def gaussian_probability(feature_data, mean, st_dev):

        variance = st_dev ** 2
        denominator = np.sqrt(2 * np.pi * variance)
        numerator = np.exp(-(feature_data - mean) ** 2 / (2 * variance))
        return numerator / denominator

    @staticmethod
    def calculate_cost(predictions, labels):
        return np.sum(np.abs(np.array(predictions) - labels))