'''

Flask_App

Author: Armin Berger
First created:  06/04/2022
Last edited:    02/05/2022

OVERVIEW:
This file seeks to deploy a pre-built ML model.
The user gives Text input to the model and the model then classifies whether
the news is reliable or not.

'''

from flask import Flask
app = Flask(__name__)

@app.route('/')

def hello_world_app():
    message = "Hello World!!"
    return message

if __name__ == "__main__":
    app.run(debug=True)