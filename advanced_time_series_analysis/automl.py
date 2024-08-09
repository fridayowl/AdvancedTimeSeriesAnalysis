import optuna
from .modeling import TimeSeriesModel, ARIMAModel, VARModel, LSTMModel, ProphetModel
from .evaluation import TimeSeriesModelEvaluator

class TimeSeriesAutoML:
    """Automated machine learning for time-series models"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model_evaluator = TimeSeriesModelEvaluator()

    def objective(self, trial):
        model_type = trial.suggest_categorical('model_type', ['arima', 'var', 'lstm', 'prophet'])
        model_params = {}

        if model_type == 'arima':
            model_params['p'] = trial.suggest_int('p', 1, 3)
            model_params['d'] = trial.suggest_int('d', 1, 2)
            model_params['q'] = trial.suggest_int('q', 1, 3)
            model = ARIMAModel(**model_params)
        elif model_type == 'var':
            model_params['lag_order'] = trial.suggest_int('lag_order', 1, 6)
            model = VARModel(**model_params)
        elif model_type == 'lstm':
            model_params['units'] = trial.suggest_int('units', 32, 128, step=32)
            model_params['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
            model = LSTMModel(**model_params)
        elif model_type == 'prophet':
            model_params['changepoint_prior_scale'] = trial.suggest_float('changepoint_prior_scale', 0.01, 0.5)
            model_params['seasonality_prior_scale'] = trial.suggest_float('seasonality_prior_scale', 1.0, 10.0)
            model = ProphetModel(**model_params)

        forecaster = TimeSeriesForecaster(model)
        forecaster.fit(self.X, self.y)
        y_pred = forecaster.predict(self.X, horizon=12)
        mape = self.model_evaluator.evaluate_out_of_sample(forecaster, self.X, self.y, self.X, self.y)['MAPE']
        return mape

    def find_best_model(self, n_trials=100):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params

    def build_model(self, best_params):
        model_type = best_params['model_type']
        if model_type == 'arima':
            return ARIMAModel(**{k: best_params[k] for k in ['p', 'd', 'q']})
        elif model_type == 'var':
            return VARModel(**{k: best_params[k] for k in ['lag_order']})
        elif model_type == 'lstm':
            return LSTMModel(**{k: best_params[k] for k in ['units', 'dropout']})
        elif model_type == 'prophet':
            return ProphetModel(**{k: best_params[k] for k in ['changepoint_prior_scale', 'seasonality_prior_scale']})