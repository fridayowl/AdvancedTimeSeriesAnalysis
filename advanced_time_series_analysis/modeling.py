import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class TimeSeriesModel(BaseEstimator, RegressorMixin):
    """Base class for time-series models"""
    def __init__(self, model_type='arima', **kwargs):
        self.model_type = model_type
        self.model = None
        self.fit_params = kwargs

    def fit(self, X, y):
        if self.model_type == 'arima':
            self.model = ARIMAModel(X, y, **self.fit_params)
        elif self.model_type == 'var':
            self.model = VARModel(X, y, **self.fit_params)
        elif self.model_type == 'lstm':
            self.model = LSTMModel(X, y, **self.fit_params)
        self.model.fit()

    def predict(self, X):
        return self.model.predict(X)

class ARIMAModel(TimeSeriesModel):
    """ARIMA time-series model"""
    def __init__(self, p=1, d=1, q=1, **kwargs):
        super().__init__(model_type='arima', p=p, d=d, q=q, **kwargs)
        self.model = ARIMA(order=(p, d, q))

class VARModel(TimeSeriesModel):
    """Vector Autoregressive (VAR) time-series model"""
    def __init__(self, lag_order=1, **kwargs):
        super().__init__(model_type='var', lag_order=lag_order, **kwargs)
        self.model = VAR(endog=kwargs['endog'], exog=kwargs['exog'], lag_order=lag_order)

class LSTMModel(TimeSeriesModel):
    """Long Short-Term Memory (LSTM) time-series model"""
    def __init__(self, units=64, dropout=0.2, **kwargs):
        super().__init__(model_type='lstm', units=units, dropout=dropout, **kwargs)
        self.model = self._build_model(kwargs['input_shape'])

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(self.units, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model