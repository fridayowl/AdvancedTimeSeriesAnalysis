import pandas as pd
import numpy as np

dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
data = {
    'date': dates,
    'variable1': np.random.randint(10, 100, size=len(dates)),
    'variable2': np.random.randint(50, 200, size=len(dates)),
    'variable3': np.random.randint(1, 50, size=len(dates))
}
df = pd.DataFrame(data)
df.to_csv('multivariate_data.csv', index=False)