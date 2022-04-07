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
import os
from flask import Flask, render_template, request

# suppress common warnings
import warnings
warnings.filterwarnings("ignore")

# import file which has all text pre processing functions
from text_pre_processing import *     # import classes and functions

### CREATE A TFIDF VECTORIZER

# set current directory
current_dir = os.getcwd()

# load in the trained vectorizer
filename = f'{current_dir}/vectorizer.pk'
vectorizer = pickle.load(open(filename, 'rb'))

### LOAD IN MODEL AND GET USER INPUT

# load the model from disk
filename = f'{current_dir}/basic_news_logistic_regression.sav'
model = pickle.load(open(filename, 'rb'))


app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def home():

    user_news_input = request.form['a']

    # ensure that the user input is in string format
    if isinstance(user_news_input, str):

        # use try and except statement to catch faulty input
        try:
            # pre-process the user input in the same manner as the training data
            user_news_input_processed = futher_process_string(user_news_input)
            user_news_input_processed = ' '.join(user_news_input_processed)
            user_news_input_vec = vectorizer.transform([user_news_input_processed])

            # prediction of our target
            prediction = model.predict(user_news_input_vec)
        except:
            print('Problem with user input')

    return render_template('after.html', data=prediction)


if __name__ == "__main__":
    app.run()
