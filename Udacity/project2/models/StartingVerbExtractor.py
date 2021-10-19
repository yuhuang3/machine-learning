import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.base import BaseEstimator, TransformerMixin

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    '''
    given a string of texts, this function does the following:
    - break the string into tokens (words)
    - get the lemma of each token
    - change the lemma to lower case and trim spaces 
    '''
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        '''
        this method does the following:
        - tokenize a string of texts into sentences 
        - for each sentense, tokenize the sentense and find the POS (Position of Speach) of each token
        - return True if any of the sentenses starts with Verb, False otherwise
        '''
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        '''
        this function does the following:
        - transforms a Pandas Series of strings into a DataFrame of True/False values,
        indicating whether or not any sentense in a string starts with Verb. 
        - one-hot-encode the resulting Series of True/False values
        '''
        X_tagged = pd.Series(X).apply(self.starting_verb)
        X_tagged_df = pd.DataFrame(X_tagged)
        dummies = pd.get_dummies(X_tagged_df[0], prefix='StartingVerb')
        return dummies
