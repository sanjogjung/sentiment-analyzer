from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
from preprocessor import Preprocess
import pandas as pd
import sys


class Test:
    def __init__(self, x_test, y_test):
        """
        initializing x_test and y_test
        :param x_test:
        :param y_test:
        """
        self.x_test = x_test
        self.y_test = y_test

    def main(self):

        """
        creating an instance of Preprocess class of preprocessor module
        applying the clean_data function to the x_test
        loading the vectorizer saved during training
        loading different models with load_models function
        saving the predictions that are made on the features
        calling metrics function to display the model metrics
        :return:
        """
        prpobj = Preprocess()
        refined_text = self.x_test.apply(prpobj.clean_data)
        tf1_old = pickle.load(open('vector.pkl', 'rb'))
        features = tf1_old.transform(refined_text)
        model_names = ['model1.pkl', 'model2.pkl', 'model3.pkl']
        self.models = self.load_models(model_names)
        self.predictions = self.prediction(features)
        self.classifiers = ['Naive-bayes', 'Decision-tree', 'Support-vectors']
        self.metrics()

    def metrics(self):
        """
        printing the different metrics for models
        confusion metrics,accuracy, classification report

        """
        for i in range(len(self.predictions)):
            print('Accuracy achieved  for ' + self.classifiers[i] + ': ')
            print(accuracy_score(self.y_test, self.predictions[i]))
            print('Confusion Matrix : \n')
            print(confusion_matrix(self.y_test, self.predictions[i]))
            print('Classification report \n')
            print(classification_report(self.y_test, self.predictions[i]))

    def prediction(self, features):
        """
        for checking the accuracy we test our model with test data
        - iterating through the models
        - predicting and appending predictions to the predictions list
        :return: predictions

        """
        predictions = []
        for i in range(len(self.models)):
            pred = self.models[i].predict(features)
            predictions.append(pred)
        return predictions

    def load_models(self, model_names):
        """
        iterating through the model names
        and appending each model to the model list
        :param model_names:
        :return: models

        """
        models = []
        for i in range(len(model_names)):
            model = pickle.load(open(model_names[i], 'rb'))
            models.append(model)
        return models


if __name__ == '__main__':
    print(sys.argv)
    filename = sys.argv[1:]
    df = pd.read_csv(filename[0])
    obj1 = Test(df['review'], df['sentiment'])
    obj1.main()




