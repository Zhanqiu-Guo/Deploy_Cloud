from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression # importing Sklearn's logistic regression's module
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)
scale=StandardScaler()
X_train = scale.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)

app = Flask(__name__)

# Load the saved model
model = joblib.load('finalized_model.sav')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    dict = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    # Get input from user and make prediction using loaded model
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction  = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='The flower is most likely to be {}'.format(dict[output]))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    app.run(debug=True, host="0.0.0.0", port=8989)