import pickle
import pandas as pd
import sys
from preprocessor import Preprocess


class Prediction:
    def __init__(self, text):
        self.text = text

    def main(self):
        """
        initializing an object of Preprocess class of preprocessor object
        loading the tfidf-vectorizer saved during training
        transforming text to extract features
        calling predict_sentiment function to predict the statement

        """
        preobj = Preprocess()
        processed_text = preobj.clean_data(text)
        tf1_old = pickle.load(open('vector.pkl', 'rb'))
        self.features = tf1_old.transform(pd.Series(processed_text))
        self.predict_sentiment()

    def predict_sentiment(self):
        """
        loading the model that we saved during training
        predicting with the given features
        printing the sentiment according the prediction

        """
        MNB = pickle.load(open('model1.pkl', 'rb'))
        pred = MNB.predict(self.features)
        if pred[0] == 0:
            print('negative')
        else:
            print('positive')


if __name__ == '__main__':
    print(sys.argv)
    text = sys.argv[1:]
    text = ' '.join(text)
    p1 = Prediction(text)
    p1.main()


