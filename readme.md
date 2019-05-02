## Sentiment Analyzer
##### 
The main purpose of this sentiment analyzer is to understand people's sentiment or opinion about the movie on the basis of their review. Here our program  understands the people's review and classifies them as positive or negative. 


### Working: 

The raw text we get as input will be preprocessed via the preprocessor module and the features will be extracted from them as tfidf scores of each unique words in the document. These features will help us to classify the text as positive or negative. 


##### The program consist of the following modules .
 - preprocessor.py :for cleaning raw text and etracting features

- trainer.py : for training our models and saving them
 
- test.py : for checking how our models performed on test data

- predict.py :for predicting from the terminal 

- api.py : for predicting from the postman

- split.py : for splitting the dataset into training and testing
 
 ### Libraries:
- Sklearn
    - TFIDF vectorizer
    - Multinomial Naive Bayes classifier
    - Decision Tree
    - Support Vector Machines
 - Pandas
 - Numpy
 - Pickle
 - Flask (for api )
    -request
 - sys
 - string
 
 
 
 
 
 ### How to run the program :
 First of all you have to train the model with a dataset
 These data sets will be automatically created for you by just running
 the following command
 
 ```
 python split.py
```
 
 For training you can just run the following command
 ```
python trainer.py train.csv
```
>Here train.csv is the file generated by the split.py script

Then you can predict sentiment in different ways
 > One can directly predict from the terminal :
 
 The corresponding script will be:
```
python predict.py the movie is bad
```
This command will return us the sentiment

> We can also predict the sentiment with postman

The command will be

``` 
python api.py
```
Then you can make a post request in postman by going to localhost:5000 and typing the review in the body part of postman in  JSON format

> {"review": "the movie is bad "}

-Then our program will return us a JSON object as a response

>{"sentiment": "negative""}

You can evaluate the models that you trained earlier .
the command will be

```
python test.py test.csv
```
> Here "test.csv" is the data set that you will use to evaluate you model


 



























 
 
 
 
 
 
 

