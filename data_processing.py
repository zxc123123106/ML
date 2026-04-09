import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer as sp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv('/Users/ian/Documents/project/src/practice_data/sensor_data.csv')
# imputer = sp(strategy='Median')
# df_imputed = imputer.fit_transform(df)

x = np.array([[1.0, 2.0], [6.0, 7.0], [3.0, 4.0]])
y = np.array(['中', '低', '高'])

# scalar = StandardScaler()
# x_scaled = scalar.fit_transform(x)
# print(x_scaled)

label = LabelEncoder()
y_encoder = label.fit_transform(y)
print(y_encoder)