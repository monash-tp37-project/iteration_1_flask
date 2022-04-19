'''

Flask_App

Author: Armin Berger
First created:  06/04/2022
Last edited:    06/04/2022

OVERVIEW:
This file seeks to deploy a pre-built ML model.
The user gives Text input to the model and the model then classifies whether
the news is reliable or not.

'''

# import the required packages
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from flask import Flask, render_template, request


### CREATE A TFIDF VECTORIZER

# load in the trained vectorizer
filename = 'vectorizer.pk'
vectorizer = pickle.load(open(filename, 'rb'))

### LOAD IN MODEL AND GET USER INPUT

# load the model from disk
filename = 'basic_news_logistic_regression.sav'
model = pickle.load(open(filename, 'rb'))

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", \
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", \
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", \
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", \
             "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", \
             "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", \
             "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", \
             "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", \
             "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", \
             "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


# function which does further string preprocessing
def process_string(text):

    # ensure that the text is in string format
    text = str(text)

    # split the string into individual tokens
    text_list = text.split(' ')

    # save a list of strings
    final_string_list = []

    for token in text_list:

        token = token.lower()

        token = token.strip(' ')

        if token not in stopwords:

            clean_word = ''

            for char in token:

                if char.isalnum():

                    clean_word = clean_word + char

            if len(clean_word) > 0:
                final_string_list.append(clean_word)

    return final_string_list

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/', methods=['POST'])

def home():

    user_news_input = request.form['a']

    # ensure that the user input is in string format
    if isinstance(user_news_input, str):

        # pre-process the user input in the same manner as the training data
        user_news_input_processed = process_string(user_news_input)
        user_news_input_processed = ' '.join(user_news_input_processed)
        user_news_input_vec = vectorizer.transform([user_news_input_processed])

        # prediction of our target
        prediction = model.predict(user_news_input_vec)

    return render_template('after.html', data=prediction)


if __name__ == "__main__":
    app.run(debug=True)
