from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class Predictor:
    def __init__(self, model_handler, true_labels, texts):
        self.model_handler = model_handler
        self.true_labels = true_labels
        self.texts = texts
        self.predictions = []

    def make_predictions(self):
        for text in self.texts:
            predicted_class_id = self.model_handler.predict(text)
            self.predictions.append(predicted_class_id)

    def calculate_metrics(self):
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(self.true_labels, self.predictions, average="binary")
        return accuracy, precision, recall, f1
