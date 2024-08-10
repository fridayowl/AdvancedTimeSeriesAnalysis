import pandas as pd
import numpy as np

dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
data = {
    'date': dates,
    'sales': np.random.randint(100, 1000, size=len(dates)),
    'temperature': np.random.uniform(0, 30, size=len(dates)),
    'promotion': np.random.choice([0, 1], size=len(dates), p=[0.7, 0.3])
}
df = pd.DataFrame(data)
df.to_csv('sales_data.csv', index=False)