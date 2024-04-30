import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from joblib import dump, load
import openpyxl

app = Flask(__name__)
loaded_model = load('GradientBoostingRegressorModel.joblib')

@app.route('/', methods = ['GET'])
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():

    # getting data
    data = dict(request.form)

    # change it to pandas DataFrame
    data["Tarix"] = [data["Tarix"]]
    data["Ma\u011faza n\u00f6mr\u0259si"] = [data["Ma\u011faza n\u00f6mr\u0259si"]]
    data["Stok Kodu"] = [data["Stok Kodu"]]
    data = pd.DataFrame(data)

    # rename columns
    data.rename(columns = {'Tarix' : 'Date',
                            'Ma\u011faza n\u00f6mr\u0259si' : 'Market Number',
                            'Stok Kodu' : 'Product Code',
                            },
                inplace = True)

    # change types of columns
    data['Date'] = pd.to_datetime(data['Date'], format = '%Y-%m-%d')
    data['Market Number'] = data['Market Number'].astype(int)
    data['Product Code'] = data['Product Code'].astype(str)

    # separate "Date" column into days and month columns
    data['Day'] = data['Date'].dt.day.astype(int)
    data['Month'] = data['Date'].dt.month.astype(int)

    # drop useless columns
    data = data.drop('Date', axis = 1)

    # load encoder
    encoder = load('encoder.joblib')

    data['Product Code'] = data['Product Code'].map(lambda x: '<unknown>' if x not in encoder.classes_ else x)
    encoder.classes_ = np.append(encoder.classes_, '<unknown>')

    # transform data point using encoder
    data['Product Code'] = encoder.transform(data['Product Code'])

    # create new DataFrame
    new_data = pd.DataFrame({'Market Number' : data['Market Number'], 
                            'Product Code' : data['Product Code'], 
                            'Day' : data['Day'],
                            'Month' : data['Month']})

    # predict new data point
    prediction = list(loaded_model.predict(new_data))
    if prediction[0] < -1:
        prediction[0] = 0
    elif -1 <= prediction[0] <= 0:
        prediction[0] += 1.1
    elif 0 < prediction[0] <= 5:
        prediction[0] += 2.5
    elif 5 < prediction[0] <= 20:
        prediction[0] += 5
    elif 20 < prediction[0] <= 30:
        prediction[0] += 9
    else:
        prediction[0] += 17

    return prediction


if __name__ == '__main__':
    app.run(debug = True) 
