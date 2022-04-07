import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import collections
import spacy
import unicodedata


english_stop_words = stopwords.words('english')

# function to remove unwanted characters from as string
def remove_char(text):
    # list of unwated charaters
    remove_characters = ["'", ';', ':', '!', "*", '/', '\\', "(", ")", ",", ".", "-", "&", "\n", '“', '@', '–', '"',
                         '+', '=', '[', ']', '?', '”']

    # loop through all unwated characters
    for character in remove_characters:
        text = text.replace(character, " ")

    return text



# function that replaces accentuated characters by their non-accentuated counterparts
# in a string
def remove_accents(text):

    text = unicodedata.normalize('NFKD', text)

    return "".join([c for c in text if not unicodedata.combining(c)])



# function to clean a string and turn it into a uniform format
# we can either keep numbers or remove them
def clean_string(text):
    text = str(text)

    text = text.lower()

    text = text.replace("'", "")

    text = remove_char(text)

    text = text.strip(' ')

    text = remove_accents(text)

    return text



# function which turns an uncleaned sting containing a number of tokens into a list of cleaned tokens
def futher_process_string(text):
    # ensure that the text is in string format
    text = str(text)

    # don't keep numbers
    if text.isnumeric() == False:

        # split the string into individual tokens
        text = text.split(' ')

        # save a list of strings
        final_string_list = []

        # loop through all tokens
        for token in text:

            # clean the string
            token = clean_string(token)

            # don't keep numbers
            if token.isnumeric() == False and token not in english_stop_words:

                # ensure that the token is not None
                if token is not None and token != '':

                    # lemmatize the token
                    token = WordNetLemmatizer().lemmatize(token)

                    # append the cleaned and lemmatized token
                    final_string_list.append(token)

        # return the the final string list
        return final_string_list



# input is a list of lists containing strings
def remove_all_non_utf8_characters(news_text_list):

    # the final cleaned list of lists
    cleaned_news_list = []

    for article in news_text_list:

        cleaned_article = []

        for word in article:

            clean_word = ''

            for char in word:

                if char.isalnum():

                    clean_word = clean_word + char

            if len(clean_word) > 0:

                cleaned_article.append(clean_word)

        cleaned_news_list.append(cleaned_article)

    # return the final list
    return cleaned_news_list


# main block to check if the code is working the way you expect
if __name__ == '__main__':
    pass                    # pass in case there is no code