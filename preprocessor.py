import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class Preprocess:
    def process_data(self, df):
        """
        cleaning the review
        extracting features
        :param Dataframe
        :return: features, output
        """
        processed_text = df['review'].apply(self.clean_data)
        features = self.feature_selection(processed_text)
        return features, df['sentiment']

    def clean_data(self, review):
        """
        removing punctuation characters from the text
        removing stopwords using the text file
        returning the refined text
        :param review:
        :return: joined tokens
        """
        remove_punc = [char for char in review if char not in string.punctuation]
        no_punc = ''.join(remove_punc)
        tokens = no_punc.split()
        with open('stopwords.txt', 'r') as f:
            stopwords = f.read().splitlines()
        refined_text = [token for token in tokens if token.lower() not in stopwords]
        return ' '.join(refined_text)

    def feature_selection(self, refined_text):
        """
        making vectorizer with ngram_range =(1,2)
        fitting and transforming the vocabs
        saving the vectorizer for further use

        :param vocabs:
        :return: tfidf_matrix
        """
        tv = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = tv.fit_transform(refined_text)
        pickle.dump(tv, open("vector.pkl", "wb"))
        return tfidf_matrix










