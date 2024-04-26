import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from joblib import dump, load
import pickle

app = Flask(__name__)
loaded_model = load('GradientBoostingRegressorModel.joblib')

@app.route('/', methods = ['GET'])
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():

    file = request.files['file']

    if file.filename == '':
        # prediction for an individual data point
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
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        data['Product Code'] = data['Product Code'].map(lambda x: '<unknown>' if x not in encoder.classes_ else x)
        encoder.classes_ = np.append(encoder.classes_, '<unknown>')

        # transform data point using encoder
        data['Product Code'] = encoder.transform(data['Product Code'])

        # create new DataFrame
        new_data = pd.DataFrame({'Market Number' : data['Market Number'], 
                                'Product Code' : data['Product Code'], 
                                'Day' : data['Day'],
                                'Month' : data['Month']})

        # predict new_data
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
        
    else:
        # prediction for file
        file_data = pd.read_excel(file)
        file_data = file_data[:1000]

        # change file_data to pandas DataFrame
        file_data = pd.DataFrame(file_data)

        # rename file_data columns
        file_data.rename(columns = {'Tarih' : 'Date',
                                    'Mağaza No' : 'Market Number',
                                    'Stok Kodu' : 'Product Code',
                                   },
                        inplace = True)
    
        # change types of columns in file_data
        file_data['Date'] = pd.to_datetime(file_data['Date'], errors = 'coerce', format = 'mixed')
        file_data['Date'] = file_data['Date'].dt.date
        file_data['Date'] = pd.to_datetime(file_data['Date'], format = '%Y-%m-%d')
        file_data['Market Number'] = file_data['Market Number'].astype(int)
        file_data['Product Code'] = file_data['Product Code'].astype(str)

        # separate "Date" column into days and month columns in file_data
        file_data['Day'] = file_data['Date'].dt.day.astype(int)
        file_data['Month'] = file_data['Date'].dt.month.astype(int)

        # drop useless columns in file_data
        file_data = file_data.drop('Date', axis = 1)

        # load encoder for file_data
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        file_data['Product Code'] = file_data['Product Code'].map(lambda x: '<unknown>' if x not in encoder.classes_ else x)
        encoder.classes_ = np.append(encoder.classes_, '<unknown>')

        # transform data point using encoder in file_data
        file_data['Product Code'] = encoder.transform(file_data['Product Code'])

        # create new DataFrame for file_data
        new_data_for_file = pd.DataFrame({'Market Number' : file_data['Market Number'], 
                                          'Product Code' : file_data['Product Code'], 
                                          'Day' : file_data['Day'],
                                          'Month' : file_data['Month']})

        # predict file_data
        prediction_for_file = list(loaded_model.predict(new_data_for_file))

        for i in range(len(prediction_for_file)):
            if prediction_for_file[i] < -1:
                prediction_for_file[i] = 0
            elif -1 <= prediction_for_file[i] <= 0:
                prediction_for_file[i] += 1.1
            elif 0 < prediction_for_file[i] <= 5:
                prediction_for_file[i] += 2.5
            elif 5 < prediction_for_file[i] <= 20:
                prediction_for_file[i] += 5
            elif 20 < prediction_for_file[i] <= 30:
                prediction_for_file[i] += 9
            else:
                prediction_for_file[i] += 17

        return prediction_for_file


if __name__ == '__main__':
    app.run(port = 3000, debug = True) 
