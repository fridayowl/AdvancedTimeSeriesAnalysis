from .modeling import TimeSeriesModel, ARIMAModel, VARModel, LSTMModel

class TimeSeriesForecaster:
    """Provides forecasting capabilities using time-series models"""
    def __init__(self, model_type='arima', **kwargs):
        if model_type == 'arima':
            self.model = ARIMAModel(**kwargs)
        elif model_type == 'var':
            self.model = VARModel(**kwargs)
        elif model_type == 'lstm':
            self.model = LSTMModel(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test, horizon=1):
        return self.model.predict(X_test, horizon=horizon)