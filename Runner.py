from service.data_service import DataService
from naive_bayes_learner import NaiveBayesLearner
from service.report_service import ReportService


class Runner:

    def __init__(self, normalization_method='z'):
        self.naive_bayes_learner = None
        self.normalization_method = normalization_method
        self.report_service = ReportService()

    def run(self, k=2):

        data = DataService.load_csv("data/wdbc.data")
        # column 1 is the id, column 2 is the label, the rest are features
        feature_data = data[:, 2:]
        labels_data = data[:, 1]

        self.naive_bayes_learner = NaiveBayesLearner()
        self.naive_bayes_learner.train(feature_data, labels_data)

        predictions = []
        for record_index, feature_record in enumerate(feature_data):
            prediction = self.naive_bayes_learner.predict(feature_record)
            predictions.append(prediction)

        self.report_service.report(data, predictions, labels_data, self.naive_bayes_learner)


if __name__ == "__main__":

    Runner(normalization_method='z').run(k=2)
