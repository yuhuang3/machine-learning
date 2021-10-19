import sys
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    this function does the following:
    - load messages dataset
    - merge datasets
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on = ['id'])
    return df


def clean_data(df):
    '''
    this function does the following:
    - create a dataframe of the 36 individual category columns
    - extract a list of new column names for categories
    - convert the strings of categories into 0/1 integers
    - concatenate the original dataframe with the new `categories` dataframe
    - drop duplicates
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories']
    categories = categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split("-")[0])

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split("-")[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # one-hot encoding
    categories_copy = categories.copy()
    for column in categories_copy:
        dummies = pd.get_dummies(categories_copy[column], prefix=column)
        categories = pd.concat([categories, dummies], sort=False, axis=1)
        categories = categories.drop([column], axis=1)
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], sort=False, axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''
    save the dataframe into a table in sqlite database
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists='replace')  


def main():
    '''
    this main function does the following:
    - extract arguments from command line
    - load data from csv files into Pandas dataframe in memory
    - clean data
    - save data into database
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()