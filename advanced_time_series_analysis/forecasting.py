class TimeSeriesForecaster:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        # Call the model's fit method with the appropriate arguments
        if y is not None:
            self.model.fit(X, y)
        else:
            self.model.fit(X)

    def predict(self, X, horizon=1):
        return self.model.predict(X, horizon)
