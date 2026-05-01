import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.px = {}
        self.py = {}
        self.smoothing = smoothing

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        labels = set(y)
        self.labels = labels
        V = len(labels)
        features = X.shape[1]
        self.py = {y_ : np.count_nonzero(y == y_) for y_ in labels}
        self.px = {feature: {label: {v : 0} for label in labels for v in (0,1)} for feature in range(features)}
        for feature in range(features):
            for label in labels:
                for u in (0,1):
                    self.px[feature][label][u] = (sum(x[feature] == u and y_ == label for x, y_ in zip(X, y)) + self.smoothing) / (self.py[label] + self.smoothing * V)

    def predict(self, X):
        preds = []
        for x in X:
            probs = {label : self.py[label] * np.prod(
                np.array([self.px[i][label][x[i]] for i in range(len(x))])   
            )
                for label in self.labels}
            pred = max(probs, key = probs.get)
            preds.append(pred)
        return np.array(preds)