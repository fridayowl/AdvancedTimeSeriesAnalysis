import pandas as pd
import numpy as np

from advanced_time_series_analysis.modeling import ARIMAModel, VARModel, LSTMModel, ProphetModel
from advanced_time_series_analysis.forecasting import TimeSeriesForecaster
from advanced_time_series_analysis.causality import CausalAnalysis
from advanced_time_series_analysis.feature_engineering import TimeSeriesFeatureEngine
from advanced_time_series_analysis.evaluation import TimeSeriesModelEvaluator
from advanced_time_series_analysis.automl import TimeSeriesAutoML

# Load data
sales_df = pd.read_csv('sales_data.csv')
sales_df['date'] = pd.to_datetime(sales_df['date'])
sales_df.set_index('date', inplace=True)

multivariate_df = pd.read_csv('multivariate_data.csv')
multivariate_df['date'] = pd.to_datetime(multivariate_df['date'])
multivariate_df.set_index('date', inplace=True)

# 1. ARIMA Model Forecasting
def arima_forecasting():
    model = ARIMAModel(p=1, d=1, q=1)
    forecaster = TimeSeriesForecaster(model)
    
    # Fit the model using past data
    forecaster.fit(sales_df[['temperature']], sales_df['sales'])
    
    # Prepare exogenous data for the forecast horizon
    X_future = sales_df[['temperature']].iloc[-30:]
    
    # Forecast the next 30 steps
    forecast = forecaster.predict(X_future, horizon=30)
    
    print("ARIMA Forecast for next 30 days:", forecast)int("ARIMA Forecast for next 30 days:", forecast)

# 2. VAR Model Forecasting
def var_forecasting():
    model = VARModel(lag_order=2)
    forecaster = TimeSeriesForecaster(model)
    forecaster.fit(multivariate_df)  # No separate target needed for VAR
    forecast = forecaster.predict(multivariate_df, horizon=30)
    print("VAR Forecast for next 30 days:", forecast)
# 3. LSTM Model Forecasting
def lstm_forecasting():
    model = LSTMModel(units=64, dropout=0.2)
    forecaster = TimeSeriesForecaster(model)
    forecaster.fit(sales_df[['temperature', 'promotion']], sales_df['sales'])
    forecast = forecaster.predict(sales_df[['temperature', 'promotion']], horizon=30)
    print("LSTM Forecast for next 30 days:", forecast)

# 4. Prophet Model Forecasting
def prophet_forecasting():
    model = ProphetModel()
    forecaster = TimeSeriesForecaster(model)
    forecaster.fit(sales_df[['temperature']], sales_df['sales'])
    forecast = forecaster.predict(sales_df[['temperature']], horizon=30)
    print("Prophet Forecast for next 30 days:", forecast)

# 5. Causal Impact Analysis
def causal_impact_analysis():
    ca = CausalAnalysis()
    intervention_time = '2021-06-01'
    summary, _ = ca.causal_impact_analysis(sales_df['sales'], intervention_time)
    print("Causal Impact Analysis Summary:", summary)

# 6. Causal Model Analysis
def causal_model_analysis():
    ca = CausalAnalysis()
    estimate = ca.causal_model_analysis(sales_df, 'promotion', 'sales', ['temperature'])
    print("Causal Model Analysis Estimate:", estimate)

# 7. Automatic Feature Engineering
def automatic_feature_engineering():
    fe = TimeSeriesFeatureEngine()
    features = fe.automatic_feature_engineering(sales_df.reset_index())
    print("Automatically Engineered Features:", features.columns)

# 8. Model Evaluation - Out of Sample
def evaluate_out_of_sample():
    model = ARIMAModel(p=1, d=1, q=1)
    forecaster = TimeSeriesForecaster(model)
    forecaster.fit(sales_df[['temperature']], sales_df['sales'])
    
    evaluator = TimeSeriesModelEvaluator()
    metrics = evaluator.evaluate_out_of_sample(forecaster, sales_df[['temperature']], sales_df['sales'], 
                                               sales_df[['temperature']], sales_df['sales'])
    print("Out of Sample Evaluation Metrics:", metrics)

# 9. Evaluate Probabilistic Forecasts
def evaluate_probabilistic_forecasts():
    model = ARIMAModel(p=1, d=1, q=1)
    forecaster = TimeSeriesForecaster(model)
    forecaster.fit(sales_df[['temperature']], sales_df['sales'])
    
    evaluator = TimeSeriesModelEvaluator()
    y_true = sales_df['sales'].values[-30:]
    y_pred = forecaster.predict(sales_df[['temperature']], horizon=30)
    y_quantiles = np.percentile(y_pred, [10, 90])
    pinball_loss = evaluator.evaluate_probabilistic_forecasts(y_true, y_pred, y_quantiles)
    print("Probabilistic Forecast Evaluation (Pinball Loss):", pinball_loss)

# 10. AutoML for Time Series
def automl_time_series():
    automl = TimeSeriesAutoML(sales_df[['temperature', 'promotion']], sales_df['sales'])
    best_params = automl.find_best_model(n_trials=50)
    print("Best Model Parameters:", best_params)

# 11. Seasonal Decomposition
def seasonal_decomposition():
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(sales_df['sales'], model='additive', period=365)
    print("Seasonal Decomposition Components:", result.seasonal[:5])

# 12. Granger Causality Test
def granger_causality():
    from statsmodels.tsa.stattools import grangercausalitytests
    gc_res = grangercausalitytests(sales_df[['sales', 'temperature']], maxlag=5, verbose=False)
    print("Granger Causality Test Results:", gc_res[1][0]['ssr_ftest'][1])

# 13. Cross-Validation for Time Series
def time_series_cross_validation():
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(sales_df):
        print("TRAIN:", train_index, "TEST:", test_index)
        break  # Just print the first split

# 14. Anomaly Detection
def anomaly_detection():
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(sales_df[['sales']])
    print("Number of Anomalies Detected:", sum(anomalies == -1))

# 15. Time Series Clustering
def time_series_clustering():
    from tslearn.clustering import TimeSeriesKMeans
    X = sales_df['sales'].values.reshape(-1, 1)
    km = TimeSeriesKMeans(n_clusters=3, metric="dtw")
    labels = km.fit_predict(X)
    print("Time Series Cluster Labels:", labels[:10])

# Run all examples
if __name__ == "__main__":
    examples = [
        arima_forecasting, var_forecasting, lstm_forecasting, prophet_forecasting,
        causal_impact_analysis, causal_model_analysis, automatic_feature_engineering,
        evaluate_out_of_sample, evaluate_probabilistic_forecasts, automl_time_series,
        seasonal_decomposition, granger_causality, time_series_cross_validation,
        anomaly_detection, time_series_clustering
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example.__name__}")
        example()
