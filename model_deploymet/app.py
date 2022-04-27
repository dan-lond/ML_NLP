#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn import preprocessing


def predict_price(Year, Mileage, State, Make, Model):
    
    # Carga LabelEncoders
    leState = preprocessing.LabelEncoder()
    leState.classes_ = np.load('leState.npy', allow_pickle=True)
    leMake = preprocessing.LabelEncoder()
    leMake.classes_ = np.load('leMake.npy', allow_pickle=True)
    leModel = preprocessing.LabelEncoder()
    leModel.classes_ = np.load('leModel.npy', allow_pickle=True)
    
    # Carga modelo
    modelo_ = XGBRegressor()
    modelo_.load_model("model_xgb.txt")
    
    datos_ = pd.DataFrame.from_dict({'Year':[int(Year)],'Mileage':[int(Mileage)],
                                     'State':[int(leState.transform([' '+State]))],
                                     'Make':[int(leMake.transform([Make]))],
                                     'Model':[int(leModel.transform([Model]))],})
    
    # Make prediction
    price = modelo_.predict(datos_)
    
    return price


app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Price Prediction API',
    description='Price Prediction API')

ns = api.namespace('predict', 
     description='Price Regressor')
   
parser = api.parser()

parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Year to be analyzed', 
    location='args')
parser.add_argument(
    'Mileage', 
    type=int, 
    required=True, 
    help='Mileage to be analyzed', 
    location='args')
parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='State to be analyzed', 
    location='args')
parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='Make to be analyzed', 
    location='args')
parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Model to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'price': fields.String,
})

@ns.route('/')
class PriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def post(self):
        args = parser.parse_args()
        
        return {
         "price": predict_price(args['Year'], args['Mileage'], args['State'], args['Make'], args['Model'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)