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

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' and file2.filename == '':
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
        return prediction
    elif file2.filename == '':
        # prediction for file1
        file_data = pd.read_excel(file1)

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
        return prediction_for_file
    else:
        # prediction for file2
        file_data_train = pd.read_excel(file2)
        old_train_data = pd.read_csv('deploying_model/satis_2023_train.csv')

        # change file_data_train to pandas DataFrame
        file_data_train = pd.DataFrame(file_data_train)
        old_train_data = pd.DataFrame(old_train_data)

        # rename file_data_train columns
        file_data_train.rename(columns = {'Tarih' : 'Date',
                                          'Mağaza No' : 'Market Number',
                                          'Stok Kodu' : 'Product Code',
                                          'SATIS MIQDARI': 'Quantity',
                                         },
                                inplace = True)
        
        old_train_data.rename(columns = {'Tarih' : 'Date',
                                         'Ma?aza No' : 'Market Number',
                                         'Stok Kodu' : 'Product Code',
                                         'SATIS MIQDARI': 'Quantity',
                                         },
                                inplace = True)
    
        # change types of columns in file_data_train
        file_data_train['Date'] = pd.to_datetime(file_data_train['Date'], errors = 'coerce', format = 'mixed')
        file_data_train['Date'] = file_data_train['Date'].dt.date
        file_data_train['Date'] = pd.to_datetime(file_data_train['Date'], format = '%Y-%m-%d')
        file_data_train['Market Number'] = file_data_train['Market Number'].astype(int)
        file_data_train['Product Code'] = file_data_train['Product Code'].astype(str)
        file_data_train['Quantity'] = file_data_train['Quantity'].astype(float)

        old_train_data['Date'] = pd.to_datetime(old_train_data['Date'], errors = 'coerce', format = 'mixed')
        old_train_data['Date'] = old_train_data['Date'].dt.date
        old_train_data['Date'] = pd.to_datetime(old_train_data['Date'], format = '%d.%m.%Y')
        old_train_data['Market Number'] = old_train_data['Market Number'].astype(int)
        old_train_data['Product Code'] = old_train_data['Product Code'].astype(str)
        old_train_data['Quantity'] = old_train_data['Quantity'].astype(float)

        # separate "Date" column into days and month columns in file_data_train
        file_data_train['Day'] = file_data_train['Date'].dt.day.astype(int)
        file_data_train['Month'] = file_data_train['Date'].dt.month.astype(int)

        old_train_data['Day'] = old_train_data['Date'].dt.day.astype(int)
        old_train_data['Month'] = old_train_data['Date'].dt.month.astype(int)

        # drop useless columns in file_data_train
        file_data_train = file_data_train.drop('Date', axis = 1)
        old_train_data = old_train_data.drop('Date', axis = 1)

        # load encoder for file_data
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        old_train_data['Product Code'] = old_train_data['Product Code'].map(lambda x: '<unknown>' if x not in encoder.classes_ else x)
        encoder.classes_ = np.append(encoder.classes_, '<unknown>')

        file_data_train['Product Code'] = file_data_train['Product Code'].map(lambda x: '<unknown>' if x not in encoder.classes_ else x)
        encoder.classes_ = np.append(encoder.classes_, '<unknown>')

        # transform data point using encoder in file_data_train
        file_data_train['Product Code'] = encoder.transform(file_data_train['Product Code'])
        old_train_data['Product Code'] = encoder.transform(old_train_data['Product Code'])

        # create new DataFrame for file_data_train
        new_data_for_file_train = pd.DataFrame({'Market Number' : file_data_train['Market Number'], 
                                          'Product Code' : file_data_train['Product Code'], 
                                          'Day' : file_data_train['Day'],
                                          'Month' : file_data_train['Month'],
                                          'Quantity' : file_data_train['Quantity']})
        
        old_train_data = pd.DataFrame({'Market Number' : old_train_data['Market Number'], 
                                          'Product Code' : old_train_data['Product Code'], 
                                          'Day' : old_train_data['Day'],
                                          'Month' : old_train_data['Month'],
                                          'Quantity' : old_train_data['Quantity']})
        
        #concat old and new data points
        combined_df = pd.concat([old_train_data, new_data_for_file_train], axis=0)

        # shuffle file_data_train
        combined_df = combined_df.sample(frac = 1)

        # split into input (x) and output (y) variables
        x_train = combined_df.drop('Quantity', axis = 1)
        y_train = combined_df['Quantity']

        loaded_model.fit(x_train, y_train)
        return 'Model trained successfully'


if __name__ == '__main__':
    app.run(port = 3000, debug = True) 
