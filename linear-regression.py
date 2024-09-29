import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('height-weight.csv')
X = df[['Weight']]
Y = df['Height']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
regression = LinearRegression(n_jobs=-1)
regression.fit(X_train_scaled, Y_train)

y_pred = regression.predict(X_test_scaled)
new_data = pd.DataFrame(scaler.transform([[72]]), columns=X_train.columns)
predicted_height = regression.predict(new_data)

print(f"Predicted height for weight 72: {predicted_height[0]}")
print(f"tested values are:{y_pred}")
mse=mean_squared_error(Y_test,y_pred)
print(mse)