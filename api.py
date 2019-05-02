from flask import Flask, jsonify, request
from preprocessor import Preprocess
import pickle
import pandas as pd
app = Flask(__name__)


@app.route('/', methods=["POST"])
def check_sent():
    """
    taking taking value from the json object given by user through postman
    processing the strings of the json object
    transforming the text to extract features
    loading the saved model and predicting
    returning the json object according to the value"""
    text = request.json['review']
    prpobj= Preprocess()
    processed_text = prpobj.clean_data(text)
    tf1_old = pickle.load(open('vector.pkl', 'rb'))
    features = tf1_old.transform(pd.Series(processed_text))
    mnb = pickle.load(open('model1.pkl', 'rb'))
    pred = mnb.predict(features)
    if pred[0] == 0:
        return jsonify({'sentiment': "negative"})
    else:
        return jsonify({'sentiment': 'positive'})


if __name__ == '__main__':
    app.run(debug=True)
