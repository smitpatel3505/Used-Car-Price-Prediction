from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('cleaned_car.csv')

@app.route('/')
def index():
    brands = sorted(car['brand'].unique())
    models = sorted(car['model'].unique())
    vehicle_ages = sorted(car['vehicle_age'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    transmission_types = car['transmission_type'].unique()
    return render_template('index.html', brands=brands, models=models, vehicle_ages=vehicle_ages,
                           fuel_types=fuel_types, transmission_types=transmission_types)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        car_name = request.form.get('car_name')
        brand = request.form.get('brand')
        model_name = request.form.get('model')
        vehicle_age = float(request.form.get('vehicle_age'))
        fuel_type = request.form.get('fuel_type')
        km_driven = float(request.form.get('km_driven'))
        transmission_type = request.form.get('transmission_type')
        mileage = 19
        prediction = model.predict(pd.DataFrame({'car_name': [car_name],'brand': [brand], 'model': [model_name], 'vehicle_age': [vehicle_age],
                                                 'km_driven': [km_driven], 'fuel_type': [fuel_type],
                                                 'transmission_type': [transmission_type], 'mileage': [mileage]}))
        print(prediction)
        return str(np.round(prediction[0], 2))
    except Exception as e:
        print("An error occurred during prediction:", e)
        return "An error occurred during prediction: " + str(e)

if __name__ == '__main__':
    app.run(debug=True)
