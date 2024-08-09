import tsfresh

class TimeSeriesFeatureEngine:
    def fill_missing_values(self, data):
        """Fill missing values in the time series data"""
        return data.fillna(method='ffill')

    def automatic_feature_engineering(self, X):
        """Automatically extract relevant time-series features using tsfresh"""
        return tsfresh.extract_features(X, column_id='id', column_kind='feature', column_value='value')
