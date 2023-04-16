#Import main library
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
#Import Flask modules
from flask import Flask, request, render_template, jsonify
import pandas as pd
#Import pickle to save our regression model
import pickle 

model_d = {'silverado 1500':17173, 'tacoma': 18496,  'a5':2509, 'colorado': 5417, 'odyssey': 13929, 'civic': 5054, 'nx': 13888, 'rx': 15935, 'x5': 20663, '320': 1182, 'q5': 14893, 'c300': 4195, 'countryman': 5870, 'camery': 4408, 'mazda3': 13036}
manufactuer_d = {'audi': 3, 'bmw': 4, 'chevrolet': 7, 'ford': 13,  'honda': 16, 'lexus': 23, 'mazda': 25, 'mercedes-benz': 26, 'mini': 28, 'toyota': 38}
drive_d = {'4wd': 0, 'fwd': 1, 'rwd': 2}
fuel_d = {'diesel': 0, 'electric': 1, 'gas': 2, 'hybrid': 3}


#Initialize Flask and set the template folder to "static"
app = Flask(__name__, template_folder='', static_folder='static')

#Open our model 
model = pickle.load(open('linearRegression_model.pkl','rb'))

#create our "home" route using the "index.html" page
@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

#Set a post method to yield predictions on page
@app.route('/', methods = ['POST'])
def estimate():
     #obtain all form values and place them in an array, convert into integers
    #int_features = [int(x) for x in request.form.values()]
    #Combine them all into a final numpy array
    #final_features = [np.array(int_features)]
    #predict the price given the values inputted by user
    

    inputManufactuer = request.form['inputManufactuer']
    inputYear = request.form['inputYear']
    inputModel = request.form['inputModel']
    inputDrive = request.form['inputDrive']
    inputOdometer = request.form['inputOdometer']
    inputFuel = request.form['inputFuel']

    features = [[inputYear, inputOdometer, model_d[inputModel], drive_d[inputDrive], fuel_d[inputFuel], manufactuer_d[inputManufactuer]]]
    df = pd.DataFrame(features, columns=['year','odometer','model','drive','fuel','manufacturer'])
    scaler = StandardScaler()
    final_features = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    prediction = model.predict(final_features)
    print(prediction)

    #Round the output to 2 decimal places
    output = round(prediction[0], 2)
    #print(request.form)
    return render_template('index.html', inputManufactuer=inputManufactuer, inputYear=inputYear, inputModel=inputModel, inputDrive=inputDrive, inputOdometer=inputOdometer, inputFuel=inputFuel, price=output)

#Run app
if __name__ == "__main__":
    app.run(debug=True)
