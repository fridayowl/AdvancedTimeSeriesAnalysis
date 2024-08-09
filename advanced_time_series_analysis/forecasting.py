class TimeSeriesForecaster:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, horizon=1):
        return self.model.predict(X, horizon)
