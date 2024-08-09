from sklearn.metrics import mean_absolute_percentage_error, mean_pinball_loss

class TimeSeriesModelEvaluator:
    def evaluate_out_of_sample(self, forecaster, X, y, X_val, y_val):
        """Evaluate the model's out-of-sample performance"""
        y_pred = forecaster.predict(X_val)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        return {'MAPE': mape}

    def evaluate_probabilistic_forecasts(self, y_true, y_pred, y_quantiles):
        """Evaluate probabilistic forecasts using mean pinball loss"""
        return mean_pinball_loss(y_true, y_pred, y_quantiles)
