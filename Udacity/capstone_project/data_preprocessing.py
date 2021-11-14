import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# must use: conda install xgboost
# import xgboost as xgb
from xgboost import XGBClassifier

# must use: conda install -c conda-forge category_encoders
from category_encoders import TargetEncoder

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical

import pickle
import mlflow
import mlflow.sklearn

from config import *

def load_data(file_name, filepath='./data'):
    '''
    This function is to load csv file as Pandas dataframe.
    '''
    filepath = filepath + '/' + file_name
    data = pd.read_csv(filepath)
    return data

def get_average(value_range):
    '''
    This function is to convert an age range [x, y] to an average age of (x+y)/2.
    '''
    if value_range in ['More than 100 Days']:
        return 105
    
    alist = value_range.split('-')
    average_value = (float(alist[0]) + float(alist[1]))/2
    average_value = int(round(average_value, 0))
    
    return average_value

def save_encoders(label_encoder, target_encoders):
    '''
       Save trained label encoder and target encoders (one per categorical feature) into pickle files.
    '''
    
    # save the label encoder into file
    if label_encoder is not None:
        pickle.dump(label_encoder, open('./model/label_encoder.pkl', 'wb'))   
    
    # save the target encoders into file
    if target_encoders is not None:
        pickle.dump(target_encoders, open('./model/target_encoders.pkl', 'wb'))  
    
    return

def load_encoders():
    '''
       Load trained label encoder and target encoders (one per categorical feature) from pickle files.
    '''
    label_encoder = None
    target_encoders = None
    
    try:
        # load the label encoder from file
        label_encoder = pickle.load(open('./model/label_encoder.pkl', 'rb'))

        # load the target encoders from file
        target_encoders = pickle.load(open('./model/target_encoders.pkl', 'rb'))
    except:
        print('load_encoders(): target_encoders.pkl does not exist for model with one-hot encoding!')
    
    return label_encoder, target_encoders

# inheriting BaseEstimator, TransformerMixin to utilize pipeline
class DataCleaning(BaseEstimator, TransformerMixin):
    '''
    this class is doing the following:
        1. drop missing data because the total number of missing data is small.
        2. 
    inheriting BaseEstimator, TransformerMixin to utilize pipeline     
    '''
    def __init__(self, for_prediction = False):
        self.for_prediction = for_prediction

    # inherited from BaseEstimator
    def fit(self, x, y=None):
        print('DataCleaning.fit ...')
        return self
    
    # inherited from TransformerMixin
    def transform(self, X):
        print('DataCleaning.transform ...')
        # drop rows with missing data
        if not isinstance(X, pd.DataFrame):
            print('DataCleaning.transform: Error - X is not a dataframe')
            return X
            
        X1 = X.copy()
        
        # if not self.for_prediction:
        #    X1 = X1.dropna(axis=0)
        X1 = X1.fillna(0)
        
        # drop columns without prediction power
        drop_columns = ['case_id'] 
        for col in drop_columns: 
            if col in X1.columns:
                X1.drop([col], axis=1, inplace=True)
        
        return X1
        
class TargetEncoding(BaseEstimator, TransformerMixin):
    '''
    this class performs target encoding:
        1. convert categorical targets/labels into numbers.
        2. convert categorical features into numbers.
        
    inheriting BaseEstimator, TransformerMixin to utilize pipeline   
    '''
    def __init__(self, label_encoding_target, label_name):
        '''
        label_encoding_target: True - use label encoder to convert label column 'Stay'
                               False - use get_average() function to covert
        label_name: label column name ('Stay')
        '''
        self.label_encoding_target = label_encoding_target
        self.label_name = label_name
        self.label_encoder = None
        self.target_encoders = dict()

    def fit(self, x, y=None):
        print('TargetEncoding.fit ...')
        return self

    def transform(self, X):
        '''
        Perform target encoding transformation
        '''
        print('TargetEncoding.transform ...')
        if not isinstance(X, pd.DataFrame):
            print('TargetEncoding.transform: Error - X is not a dataframe')
            return X
        
        X1 = X.copy()
        
        # Step 1: get categorical feature names
        categorical_columns = []
        for idx, col in enumerate(X1.columns):
            # dtypes identify datatype of each column. idx identify each column.
            if X1.dtypes[idx] == object:
                if col != self.label_name:
                    categorical_columns.append(col)
                    
        # Step 2: convert the label column into numbers required by Target Encording.
        # two options are provided below.
        if self.label_encoding_target:           
            self.label_encoder = LabelEncoder()
            X1.loc[:,self.label_name] = self.label_encoder.fit_transform(X1[self.label_name])
            
            # save label encoder into file
            save_encoders(self.label_encoder, None)
        else:
            X1.loc[:, self.label_name] = X1[self.label_name].apply(lambda x: get_average(x))

        # Step 3: convert the values of categorical features in each column into numbers by taking 
        # the average of corresponding aggregated label values:
        for col in categorical_columns:
            # must create a new target encode object (TargetEncoder) for each categorical feature.
            encoder = TargetEncoder() 
            # X1.loc[:, col] = encoder.fit_transform(X1[col], X1[self.label_name])
            encoder.fit(X1[col], X1[self.label_name])
            X1.loc[:, col] = encoder.transform(X1[col])
            self.target_encoders[col] = encoder # save trained target encoders for deployment
            
        # save target eocoders into file
        save_encoders(None, self.target_encoders)

        return X1
        
class TargetEncodingForPrediction(BaseEstimator, TransformerMixin):
    '''
    this class performs target encoding:
        1. convert categorical features into numbers in deployment.
        
    inheriting BaseEstimator, TransformerMixin to utilize pipeline   
    '''
    def __init__(self, target_encoders):
        '''
        target_encoders: This is a dictionary of target encoders:
                         categorical feature name 'Age' --> trained target encoder for 'Age' column
        '''
        self.target_encoders = target_encoders

    def fit(self, x, y=None):
        print('TargetEncodingForPrediction.fit ...')
        return self

    def transform(self, X):
        print('TargetEncodingForPrediction.transform ...')
        if not isinstance(X, pd.DataFrame):
            print('TargetEncodingForPrediction.transform: Error - X is not a dataframe')
            return X
        
        X1 = X.copy()
        
        # Step 1: get categorical feature names
        categorical_columns = []
        for idx, col in enumerate(X1.columns):
            # dtypes identify datatype of each column. idx identify each column.
            if X1.dtypes[idx] == object:
                categorical_columns.append(col)
                    
        # Step 2: convert the values of categorical features in each column into numbers by taking 
        # the average of corresponding aggregated label values:
        for col in categorical_columns:
            # find the target encode object for this categorical column.
            encoder = self.target_encoders[col]
            X1.loc[:, col] = encoder.transform(X1[col])

        return X1
        
class OneHotEncoding(BaseEstimator, TransformerMixin):
    '''
    this class performs one-hot encoding:
        1. convert categorical targets/labels into numbers.
        2. convert categorical features into numbers.
        
    inheriting BaseEstimator, TransformerMixin to utilize pipeline 
    '''
    def __init__(self, label_name, for_prediction=False):
        '''
        label_name: label column name ('Stay')
        '''
        self.label_name = label_name
        self.label_encoder = None
        self.for_prediction = for_prediction

    def fit(self, x, y=None):
        print('OneHotEncoding.fit ...')
        return self

    def transform(self, X):
        print('OneHotEncoding.transform ...')
        if not isinstance(X, pd.DataFrame):
            print('OneHotEncoding.transform: Error - X is not a dataframe')
            return X
        
        X1 = X.copy()
                    
        # Step 1: convert the label column into numbers required by deep learning.
        # This step must be done first to avoid one-hot encoding this label column 
        # by get_dummies().
        if not self.for_prediction:
            self.label_encoder = LabelEncoder() # save LabelEncoder for later reverse conversion.
            X1.loc[:,self.label_name] = self.label_encoder.fit_transform(X1[self.label_name])
            # convert label_dl integers into numpy array of vectors, e.g., [[1. 0, 0, ..., 0],...]
            # label_dl = to_categorical(label_dl)
            
            # save label encoder into file
            save_encoders(self.label_encoder, None)
      
        # Step 2 (Improvement): convert Age range column into numbers
        X1.loc[:,'Age'] = X1['Age'].apply(lambda x: get_average(x))

        # Step 3: one-hot encode the values of categorical features in X1
        X1 = pd.get_dummies(X1)

        return X1
        
class FeatureNormorlization(BaseEstimator, TransformerMixin):
    '''
    this class performs one-hot encoding:
        1. convert categorical targets/labels into numbers.
        2. convert categorical features into numbers.
        
    inheriting BaseEstimator, TransformerMixin to utilize pipeline 
    '''
    def __init__(self):
        pass

    def fit(self, x, y=None):
        print('FeatureNormorlization.fit ...')
        return self

    def transform(self, X):
        print('FeatureNormorlization.transform ...')
        if not isinstance(X, pd.DataFrame):
            print('FeatureNormorlization.transform: Error - X is not a dataframe')
            return X
        
        X1 = X.copy()
                    
        # normalize feature's conponent for deep learning.
        # normalize numeric columns
        for idx, col in enumerate(X1.columns):
             if X1.dtypes[idx] != object and col != 'Stay':
                col_data = X1[col]
                mean = col_data.mean()
                col_data -= mean
                std = col_data.std()
                col_data /= std
                X1.loc[:, col] = col_data

        return X1
        
def target_encoding_preprocessing(train_data, label_encoding_target, label_name='Stay'):
    '''
        label_encoding_target: True - use label encoder to conver label column
    
        This process performs two things and is used for Random Forest Model:
        1. data cleaning.
        2. target encoding.
    '''
    target_encoder = TargetEncoding(label_encoding_target, label_name)
    
    if USE_DEEP_LEARNING_WITH_TARGET_ENCODING:
        target_encoding_pipeline = Pipeline([
            ('data_cleaning', DataCleaning()),
            ('target_encoding', target_encoder),
            ('feature_normorlization', FeatureNormorlization())
            ])        
    else:
        target_encoding_pipeline = Pipeline([
            ('data_cleaning', DataCleaning()),
            ('target_encoding', target_encoder)
            ])
    
    train_data_transformed = target_encoding_pipeline.fit_transform(train_data)
    
    # Find the correlations between features and labels
    '''
    Hospital_type_code                   0.082574
    City_Code_Hospital                   0.006626
    Hospital_region_code                 0.012746
    Available Extra Rooms in Hospital   -0.121564
    Department                           0.037188
    Ward_Type                            0.193062
    Ward_Facility_Code                   0.084282
    Bed Grade                            0.025056
    patientid                            0.000932
    City_Code_Patient                   -0.009779
    Type of Admission                    0.092835
    Severity of Illness                  0.126883
    Visitors with Patient                0.537476
    Age                                  0.097167
    Admission_Deposit                   -0.052092
    Stay                                 1.000000
    '''
    print('\nFind the correlations between each feature and label (Stay):\n')
    print(train_data_transformed[train_data_transformed.columns[0:]].corr()['Stay'][:])
       
    # Shuffle dataset to prevent artifical data patterns
    train_data_transformed = train_data_transformed.sample(frac=1)
    
    # Separate label column from feature columns
    y = train_data_transformed[label_name].values # convert Pandas Series into numpy array
    train_data_transformed.drop([label_name], axis=1, inplace=True)
    X = train_data_transformed.values # convert Pandas dataframe into numpy array
    feature_names = train_data_transformed.columns # it is needed to check feature importance.

    # Split dataset into training and testing subsets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_names, target_encoder.target_encoders, target_encoder.label_encoder

def target_encoding_preprocessing_for_prediction(test_data, target_encoders):
    '''
        test_data: dataframe without label column.
        
        This process performs two things and is used for deployment (prediction).
        1. data cleaning.
        2. target encoding.
    '''
    target_encoder = TargetEncodingForPrediction(target_encoders)
    
    if USE_DEEP_LEARNING_WITH_TARGET_ENCODING:
        target_encoding_pipeline = Pipeline([
            ('data_cleaning', DataCleaning(for_prediction=True)),
            ('target_encoding', target_encoder),
            ('feature_normorlization', FeatureNormorlization())
            ])        
    else:
        target_encoding_pipeline = Pipeline([
            ('data_cleaning', DataCleaning(for_prediction=True)),
            ('target_encoding', target_encoder)
            ])
    
    test_data_transformed = target_encoding_pipeline.fit_transform(test_data)
       
    X = test_data_transformed.values # convert Pandas dataframe into numpy array of features
    
    return X

def onehot_encoding_preprocessing(train_data, label_name='Stay'):
    '''
        This process performs two things and is used for Deep Learning Model:
        1. data cleaning.
        2. target encoding.
        
    '''
    # Create OneHotEncoding object outside of the Pipeline 
    # for convenience in getting label_encoder from the object.
    onehot_encoder = OneHotEncoding(label_name)
    onehot_encoding_pipeline = Pipeline([
            ('data_cleaning', DataCleaning()),
            ('feature_normorlization', FeatureNormorlization()),
            ('onehot_encoding', onehot_encoder)
            ])
    
    train_data_transformed = onehot_encoding_pipeline.fit_transform(train_data)
       
    # Shuffle dataset to prevent artifical data patterns
    train_data_transformed = train_data_transformed.sample(frac=1)
    
    # Separate label column from feature columns
    y = train_data_transformed[label_name].values # convert Pandas Series into numpy array
    train_data_transformed.drop([label_name], axis=1, inplace=True)
    X = train_data_transformed.values # convert Pandas dataframe into numpy array
    # The featue names can be used to check feature importance for Random Forest model.
    feature_names = train_data_transformed.columns 

    # Split dataset into training and testing subsets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_names, onehot_encoder.label_encoder


def onehot_encoding_preprocessing_for_prediction(test_data):
    '''
        test_data: dataframe without label column.
        
        This process performs two things and is used for deployment (prediction).
        1. data cleaning.
        2. target encoding.
    '''
    
    if USE_DEEP_LEARNING_WITH_ONEHOT_ENCODING:
        onehot_encoding_pipeline = Pipeline([
            ('data_cleaning', DataCleaning(for_prediction=True)),
            ('feature_normorlization', FeatureNormorlization()),
            ('onehot_encoding', OneHotEncoding(label_name=None, for_prediction=True))
            ])        
    else:
        onehot_encoding_pipeline = Pipeline([
            ('data_cleaning', DataCleaning(for_prediction=True)),
            ('onehot_encoding', OneHotEncoding(label_name=None, for_prediction=True))
            ])
    
    test_data_transformed = onehot_encoding_pipeline.fit_transform(test_data)
       
    X = test_data_transformed.values # convert Pandas dataframe into numpy array of features
    
    return X







