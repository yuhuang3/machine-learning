import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import numpy as np
# from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

import pickle

from StartingVerbExtractor import StartingVerbExtractor

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    '''
    this function does the following:
    - load the data into dataframe from database
    - extract the message column as feature
    - extract the category columns as targets/labels
    - return feature, targets/labels, and category names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(
        'DisasterResponseTable',
        con=engine
    )

    X = df['message'].values
    y = df.loc[:, 'related_0' : 'direct_report_1'].values
    
    category_names = df.loc[:, 'related_0' : 'direct_report_1'].columns
    
    return X, y, category_names


def tokenize(text):
    '''
    this function does the following:
    - replace URL with the hard coded string of "urlplaceholder"
    - tokenize text into words 
    - lemmatize tokens, change to lower case, and trim spaces
    - return the resulting clean tokens
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
    
def build_model():
    '''
    this function does the following:
    - build a machine learning pipeline 
    - set up hyper-parameters tuning options
    - create and return a GridSearchCV object
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    
    # set up hyper-parameters tuning options
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75),
        'features__text_pipeline__vect__max_features': (None, 5000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__n_jobs': [-1],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1}
        )
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names): 
    '''
    this function does the following:
    - use the provided machine learning pipeline (i.e., model) to predct categories of messages
    - print out the model performance evaluation results
    '''
    # predicting results using testing data
    y_pred = model.predict(X_test)
    
    # evaluating model performance
    labels_list = list(range(0,72))
    for col, target_name in enumerate(category_names):  
        print(classification_report(y_test[:,col], y_pred[:, col], labels=[labels_list[col]]))


def save_model(model, model_filepath):
    '''
    save the model (machine learning pipeline) into Python pickle file
    '''
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

def main():
    '''
    this main function does the following:
    - get arguments (database file path and model file path) from command line
    - get features, targets (categories), and catgory names from database
    - divide features and targets into two parts: one for model training and the other for model testing
    - build a GridSearchCV object
    - train the GridSearchCV object
    - get the best model (i.e., machine learning pipeline) from the trained GridSearchCV object
    - evaluate the model performance
    - save the model into Python pickle file
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        cv = build_model()
        
        print('Training model...')
        cv.fit(X_train, Y_train)
        
        print('Evaluating model...')
        model = cv.best_estimator_
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()