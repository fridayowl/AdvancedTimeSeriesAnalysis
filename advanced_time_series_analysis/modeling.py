import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet

class TimeSeriesModel:
    """Base class for all time-series models"""
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.fit_params = kwargs

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X, horizon=1):
        raise NotImplementedError


class ARIMAModel(TimeSeriesModel):
    """ARIMA model for time-series forecasting"""
    def __init__(self, p=1, d=1, q=1, **kwargs):
        super().__init__(model_type='arima', **kwargs)
        self.order = (p, d, q)
        self.model = None

    def fit(self, X, y):
        """Fit the ARIMA model to the data"""
        endog = y  # ARIMA is typically for univariate series
        exog = X if X is not None else None  # ARIMA can take exogenous variables
        self.model = ARIMA(endog, exog=exog, order=self.order)
        self.model = self.model.fit()

    def predict(self, X=None, horizon=1):
        """Make a forecast"""
        forecast = self.model.forecast(steps=horizon, exog=X)
        return forecast
    
class VARModel:
    def __init__(self, lag_order):
        self.lag_order = lag_order
        self.model = None

    def fit(self, X):
        self.model = VAR(X).fit(self.lag_order)

    def predict(self, X, horizon):
        # Ensure you pass the correct data for forecasting
        forecast = self.model.forecast(X.values[-self.lag_order:], steps=horizon)
        return forecast   
class LSTMModel(TimeSeriesModel):
    """LSTM-based time-series model"""
    def __init__(self, units=50, dropout=0.2):
        super().__init__(model_type='lstm', units=units, dropout=dropout)
        self.model = Sequential([
            LSTM(units=units, return_sequences=True, input_shape=(None, 1)),
            Dropout(dropout),
            LSTM(units=units),
            Dropout(dropout),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X, y, epochs=20, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X, horizon=1):
        return self.model.predict(X)


class ProphetModel(TimeSeriesModel):
    """Facebook Prophet time-series model"""
    def __init__(self, **kwargs):
        super().__init__(model_type='prophet', **kwargs)
        try:
            self.model = Prophet(**self.fit_params)
        except AttributeError as e:
            if 'np.float_' in str(e):
                # Handle the issue by using float64
                np.float_ = np.float64
                self.model = Prophet(**self.fit_params)
            else:
                raise e

    def fit(self, X, y):
        df = X.copy()
        df['y'] = y
        self.model.fit(df)

    def predict(self, X, horizon=1):
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        return forecast['yhat'].values[-horizon:]


# Example usage:
# arima_model = ARIMAModel(p=1, d=1, q=1)
# arima_model.fit(X_train, y_train)
# predictions = arima_model.predict(X_test, horizon=10)
