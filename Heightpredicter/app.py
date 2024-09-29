from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('height-weight.csv')
X = df[['Weight']]
Y = df['Height']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regression = LinearRegression(n_jobs=-1)
regression.fit(X_train_scaled, Y_train)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting height based on input weight
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            weight = float(request.form['weight'])
            # Transform the input data
            new_data = pd.DataFrame(scaler.transform([[weight]]), columns=X.columns)
            predicted_height = regression.predict(new_data)[0]
            return render_template('index.html', prediction_text=f"Predicted height for weight {weight}: {predicted_height:.2f}")
        except ValueError:
            return render_template('index.html', prediction_text="Invalid input. Please enter a valid number.")

if __name__ == "__main__":
    app.run(debug=True)
