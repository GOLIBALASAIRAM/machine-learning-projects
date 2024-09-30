from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app=Flask(__name__)

df = pd.read_csv('salary_MR.csv')
df.head()

X = df.iloc[:, :-1]  
Y = df.iloc[:, -1]   

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  

regression = LinearRegression(n_jobs=-1)
regression.fit(X_train, Y_train)

@app.route("/")
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict(): 
    if request.method=='POST':
        try:
            age=float(request.form['age'])
            experience=float(request.form['experience'])
            prediction = regression.predict(scaler.transform([[age, experience]]))
            y_new= round(prediction[0], 2)
            return render_template('index.html', prediction_text=f"Predicted salary for age {age} and experience {experience} years: {y_new:.2f}")
        except ValueError:
            return render_template('index.html', prediction_text="Invalid input. Please enter a valid number.") 

if __name__ == "__main__":
    app.run(debug=True)
