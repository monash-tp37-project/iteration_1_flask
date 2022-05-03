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

# import the required packages
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from flask import Flask, render_template, request
# libraries for keywords
import gensim
from summa import keywords
import Levenshtein


### CREATE A TFIDF VECTORIZER

# load in the trained vectorizer
filename = 'tokenizer_150k_512_10k_fake'
tokenizer = BertTokenizer.from_pretrained(filename)

### LOAD IN MODEL AND GET USER INPUT

# load the model from disk
filename = 'model_150k_512_16epoch_10k_fake'
model = BertForSequenceClassification.from_pretrained(filename)

# collection of stopwords we want to remove from user input text
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

# function which removes keywords that are too similar
def remove_duplicates(keywords_all):

    words_drop = []

    keywords_all_loop = keywords_all.copy()

    for word in keywords_all:

        keywords_all_loop.remove(word)

        for it_word in keywords_all_loop:

            sim = Levenshtein.ratio(word, it_word)

            if sim > 0.5 and word != it_word:
                words_drop.append(it_word)

    keywords_all = [x for x in keywords_all if x not in set(words_drop)]

    keywords_all = ', '.join(keywords_all)

    return keywords_all


# Remove stopwords and remove words with 2 or less characters
def preprocess(text):

    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in stopwords:
            result.append(token)

    joined_results = ' '.join(result)

    return joined_results

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/', methods=['POST'])

def home():

    user_news_input = request.form['a']

    # pre-process the user input in the same manner as the training data
    user_news_input_processed = preprocess(user_news_input)
    user_news_input_vec = tokenizer(user_news_input_processed, padding=True, truncation=True, return_tensors="pt")

    # prediction of our target
    with torch.no_grad():

        # make predictions
        output = model(**user_news_input_vec)

        # using a softmax activation function get discrete predictions
        predictions = F.softmax(output.logits, dim=1)
        labels = int(torch.argmax(predictions, dim=1))

        # using a sigmoid function get continues predictions
        prediction = torch.sigmoid(predictions).tolist()[0]
        prediction = [round(prediction[0],2),round(prediction[1],2)]

    # turn the given text input into keywords
    TR_keywords = keywords.keywords(user_news_input_processed, scores=True)
    top_keywords = [x[0] for x in TR_keywords[0:5]]
    top_keywords = remove_duplicates(top_keywords)

    # return the final data given to the "after" page
    pred_tup = prediction + [top_keywords]

    return render_template('after.html', data=pred_tup)

if __name__ == "__main__":
    app.run(debug=True)
