import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class TimeSeriesModelEvaluator:
    """Provides evaluation and diagnostic tools for time-series models"""
    def evaluate_out_of_sample(self, model, X_train, y_train, X_test, y_test):
        """Evaluate the model's out-of-sample performance"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        return {'MSE': mse, 'MAE': mae, 'MAPE': mape}

    def rolling_window_evaluation(self, model, X, y, window_size=12, step_size=1):
        """Perform rolling window evaluation of the time-series model"""
        metrics = []
        for i in range(0, len(X) - window_size, step_size):
            X_train, y_train = X[i:i+window_size], y[i:i+window_size]
            X_test, y_test = X[i+window_size:i+window_size+1], y[i+window_size:i+window_size+1]
            metrics.append(self.evaluate_out_of_sample(model, X_train, y_train, X_test, y_test))
        return metrics