from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import pickle
import preprocessor
import sys
import pandas as pd


class Train:
    def __init__(self, x_train, y_train):
        """
        initializing training data
        defining different classifiers
        """
        self.x_train = x_train
        self.y_train = y_train
        self.clf1 = MultinomialNB()
        self.clf2 = tree.DecisionTreeClassifier()
        self.clf3 = svm.SVC(kernel='linear')

    """
    fitting the model on the training dataset
    saving the corresponding model
    
    """
    def train_model(self):
        self.clf1.fit(self.x_train, self.y_train)
        pickle.dump(self.clf1, open('model1.pkl', 'wb'))

        self.clf2.fit(self.x_train, self.y_train)
        pickle.dump(self.clf2, open('model2.pkl', 'wb'))

        self.clf3.fit(self.x_train, self.y_train)
        pickle.dump(self.clf1, open('model3.pkl', 'wb'))


if __name__ == "__main__":
    print(sys.argv)
    file_name = sys.argv[1:]
    obj = preprocessor.Preprocess()
    df=pd.read_csv(file_name[0])
    features, sentiment = obj.process_data(df)
    x_train, y_train = features, sentiment
    t1 = Train(x_train, y_train)
    t1.train_model()
