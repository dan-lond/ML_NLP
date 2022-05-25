#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from joblib import load


def predict_genres(Movie):
    
    # Carga modelo
    modelo_ = load('LogReg_pipeline.joblib')
    
     
    # Make prediction
    genres_name = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
           'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical',
           'Mystery', 'News', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    genres_description = []
    genres = ((modelo_.predict_proba([Movie]) >= 0.5)*1)[0]
    if sum(genres) > 0:
        for i in range(len(genres_name)):
            if genres[i] == 1:
                genres_description.append(genres_name[i])
    else:
        genres_description = ['No matches in Generes']
    
    return genres_description


app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Movie Generes Prediction API',
    description='Movie Generes Prediction API')

ns = api.namespace('predict', 
     description='Movie Generes Classifier')
   
parser = api.parser()

parser.add_argument(
    'Movie', 
    type=str, 
    required=True, 
    help='Movie to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'Generes': fields.String,
})


@ns.route('/')
class GenresApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def post(self):
        args = parser.parse_args()
        
        return {
         "Generes": predict_genres(args['Movie'])
        }, 200
        
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)