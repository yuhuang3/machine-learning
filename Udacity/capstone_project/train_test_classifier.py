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
from data_preprocessing import *

def build_rf_model(X_train, y_train):
    '''
        This function is to build a random forest model to use grid search. 
    '''
    
    # Create a Random forest model 
    clf = RandomForestClassifier()
    
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'max_depth': [20], # best 20, tried 10, 25, 30, 50, None -- good 20
        'n_estimators': [230] # best 200,210, 220, 230 tried 100, 120,235, 240, 250 -- good 200
    }

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = clf, 
                               param_grid = param_grid, 
                               cv = 10, 
                               verbose = 3)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    print('Best hyper-parameters:\n', grid_search.best_params_)

    # get the best model
    best_model = grid_search.best_estimator_
    
    save_model(best_model, './model/rf_model.pkl')

    return best_model


def build_xgboost_model(X_train, y_train):
    '''
        This function is to build a xgboost model to use grid search. 
    '''
    
    # Create a Random forest model 
    clf = XGBClassifier(use_label_encoder=False)
    
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'max_depth': [6], # tried 20, default: 6
        'eval_metric': ['auc']
    }

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = clf, 
                               param_grid = param_grid, 
                               cv = 10, 
                               verbose = 3)
    
    # Fit the grid search to the data
    y_train = [int(x) for x in y_train] # XGBoost does not support label encoded numbers
    grid_search.fit(X_train, y_train)
    
    print('Best hyper-parameters:\n', grid_search.best_params_)

    # get the best model
    best_model = grid_search.best_estimator_
    
    save_model(best_model, './model/xgboost_model.pkl')

    return best_model

def build_deeplearning_model(X_train, y_train):
    '''
        This function is to build a MLP model with cross validation. 
    '''
    
    # Create a Deep Learning model 
    
    input_size = X_train.shape[1]
    
    if USE_DEEP_LEARNING_WITH_ONEHOT_ENCODING:
        dl_model = Sequential()
        dl_model.add(Dense(64, input_dim = input_size, activation = 'relu'))
        dl_model.add(Dropout(0.5))
        dl_model.add(Dense(32, activation = 'relu'))
        dl_model.add(Dense(16, activation = 'relu'))
        dl_model.add(Dense(11, activation='softmax'))
    else:
        dl_model = Sequential()
        dl_model.add(Dense(32, input_dim = input_size, activation = 'relu'))
        dl_model.add(Dropout(0.5))
        dl_model.add(Dense(16, activation = 'relu'))
        dl_model.add(Dense(16, activation = 'relu'))
        dl_model.add(Dense(11, activation='softmax'))
    
    #
    # categorical_crossentropy can only accept one-hot encoded labels.
    # So need to convert label integers into numpy array of vectors, e.g., [[1. 0, 0, ..., 0],...]
    # See OneHotEncoding class for details.
    #
    # dl_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #
    # sparse_categorical_crossentropy allows integer labels
    #
    dl_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    dl_model.summary()
    
    # divide training data into two: one for training and one for validation
    X_train_data, X_eval_data, y_train_data, y_eval_data = \
        train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    history = dl_model.fit(X_train_data, 
                           y_train_data, 
                           batch_size=15, 
                           epochs=4, 
                           validation_data=(X_eval_data, y_eval_data))
    
    if USE_DEEP_LEARNING_WITH_ONEHOT_ENCODING:   
        save_model(dl_model, './model/dl_onehot_model.pkl')
    else:
        save_model(dl_model, './model/dl_target_encoding_model.pkl')
                     
    return dl_model, history

def evaluate_rf_model(clf, feature_names, label_encoder, X_test, y_test, figsize=(15,15)):
    '''
        This function is doing the following:
        1. pridict and caculate the accuracy score.
        2. draw confusion matrix diagram.
        3. find and display feature importance figure.
        
    '''
    y_predict = clf.predict(X_test)
    
    acc_score = accuracy_score(y_test, y_predict)
    
    print('Accuracy score: ', acc_score)
    # this function does the same thing as above the two function calls.
    # clf.score(X_test, y_test) 
    
    label_values = label_encoder.inverse_transform(clf.classes_)
    cm = confusion_matrix(y_test, y_predict, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_values) # clf.classes_)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax)
    
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    # feature_names = [f"feature {i}" for i in range(X.shape[1])]
    # feature_names = train_data_target_encoding.columns.tolist()
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    
    return acc_score

def evaluate_xgboost_model(clf, feature_names, label_encoder, X_test, y_test):
    '''
        This function is doing the following:
        1. pridict and caculate the accuracy score.
        2. draw confusion matrix diagram.
        3. find and display feature importance figure.   
    '''
    y_predict = clf.predict(X_test)
    
    acc_score = accuracy_score(y_test, y_predict)
    print('Accuracy score: ', acc_score)
    # this function does the same thing as above the two function calls.
    # clf.score(X_test, y_test) 
    
    label_values = label_encoder.inverse_transform(clf.classes_)
    cm = confusion_matrix(y_test, y_predict, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_values) # clf.classes_)
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(ax=ax)
    
    return acc_score 

def evaluate_dl_model(dl_model, history, label_encoder, X_test, y_test):
    '''
        This function is doing the following:
        1. pridict and caculate the accuracy score.
        2. draw confusion matrix diagram.
        3. find and display feature importance figure.
        
    '''

    y_pred = dl_model.predict(X_test)
    # convert deep learning model (MLP) output vector [a0, a1, ..., a10] to the index with the max value
    # y_pred_labels return indices, each sample (row) has a index, like 0, 1, ..., 10
    y_pred_labels = np.apply_along_axis(np.argmax, 1, y_pred) 
    # convert above indices 0, 1, ..., 10 back into ranges, like "21-30", ...
    # label_values = label_encoder.inverse_transform(y_pred_labels)

    test_loss, test_accuracy = dl_model.evaluate(X_test, y_test)
    print('loss: ', test_loss, 'accuracy:', test_accuracy)

    # Ploting the traning and valitation loss  
    
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.clf()
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    
    plt.plot(epochs, acc_values, 'bo',label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # confusion matrix
    label_classes = list(range(11))
    label_classes_range = label_encoder.inverse_transform(label_classes)
    # labels=label_classes: 0, 1, .., 10
    cm = confusion_matrix(y_test, y_pred_labels, labels=label_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_classes_range) # clf.classes_)
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(ax=ax)
    
    return test_loss, test_accuracy


def save_model(model, model_filepath):
    '''
       Save trained model into pickle file.
    '''
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))
    
    return

def load_model():
    '''
      load model from pickle file.
    '''
    # load the model from disk
    if USE_RANDOM_FOREST:
        loaded_model = pickle.load(open('./model/rf_model.pkl', 'rb'))
    elif USE_XGBOOST:
        loaded_model = pickle.load(open('./model/xgboost_model.pkl', 'rb'))
    elif USE_DEEP_LEARNING_WITH_ONEHOT_ENCODING:  
        loaded_model = pickle.load(open('./model/dl_onehot_model.pkl', 'rb'))
    else: # USE_DEEP_LEARNING_WITH_TARGET_ENCODING
        loaded_model = pickle.load(open('./model/dl_target_encoding_model.pkl', 'rb'))
        
    return loaded_model

def mlFlow():
    '''
    Training model in mlflow
    '''
    np.random.seed(42)
    
    target_encoders = None
    label_encoder = None
    
    experiment_name = 'Capstone Project'
            
    mlflow.set_experiment(experiment_name)
    
    if USE_RANDOM_FOREST:
        run_name = 'Random Forest'
    elif USE_XGBOOST:
        run_name = 'XGBoost'
    elif USE_DEEP_LEARNING_WITH_TARGET_ENCODING:
        run_name = 'MLP with Target Encoding'
    else:
        run_name = 'MLP with One-Hot Encoding'
    
    with mlflow.start_run(run_name=run_name):

        #
        # Load training data
        #
        print('\nLOAD TRAINING DATA: train_data.csv\n')
        train_data = load_data('train_data.csv')
        
        #
        # Data Preprocessing
        #
        print('\nDATA PROCESSING: \n')
        if USE_RANDOM_FOREST or USE_XGBOOST or USE_DEEP_LEARNING_WITH_TARGET_ENCODING:
            # data preprocessing using target encoding
            X_train, X_test, y_train, y_test, feature_names, target_encoders, label_encoder = \
            target_encoding_preprocessing(train_data, label_encoding_target=True)
        if USE_DEEP_LEARNING_WITH_ONEHOT_ENCODING:
            # data preprocessing using onehot encoding
            X_train, X_test, y_train, y_test, feature_names, label_encoder = \
                onehot_encoding_preprocessing(train_data)
    
        #
        # Model training
        #
        print('\nMODEL TRAINING: \n')
        if USE_RANDOM_FOREST:
            # Build random forest model using GridSearch
            mlflow.log_param('model', 'Random Forest')
            mlflow.log_param('n_estimators', 230)
            mlflow.log_param('max_depth', 20)
            mlflow.log_param('data preprocessing method', 'target_encoding_preprocessing')
            mlflow.log_param('label encoding method', 'LabelEncoder')
            
            clf = build_rf_model(X_train, y_train)
            
        if USE_XGBOOST:
            # Build xgboost model using GridSearch
            mlflow.log_param('model', 'XGBoost')
            mlflow.log_param('max_depth', 6)
            mlflow.log_param('eval_metric', 'auc')
            mlflow.log_param('data preprocessing method', 'target_encoding_preprocessing')
            mlflow.log_param('label encoding method', 'LabelEncoder')
            
            clf = build_xgboost_model(X_train, y_train)
            
        if USE_DEEP_LEARNING_WITH_ONEHOT_ENCODING:
            mlflow.log_param('model', 'Deep Learning - MLP')
            mlflow.log_param('MLP layers', '64, dropout 0.5, 32, 16, 11')
            mlflow.log_param('loss', 'sparse_categorical_crossentropy')
            mlflow.log_param('optimizer', 'rmsprop')
            mlflow.log_param('metrics', 'accuracy')
            mlflow.log_param('data preprocessing method', 'onehot_encoding_preprocessing')
            mlflow.log_param('label encoding method', 'LabelEncoder')
            
            clf, history = build_deeplearning_model(X_train, y_train)
            
        if USE_DEEP_LEARNING_WITH_TARGET_ENCODING:
            mlflow.log_param('model', 'Deep Learning - MLP')
            mlflow.log_param('MLP layers', '32, dropout 0.5, 16, 16, 11')
            mlflow.log_param('loss', 'sparse_categorical_crossentropy')
            mlflow.log_param('optimizer', 'rmsprop')
            mlflow.log_param('metrics', 'accuracy')
            mlflow.log_param('data preprocessing method', 'target_encoding_preprocessing')
            mlflow.log_param('label encoding method', 'LabelEncoder')
            
            clf, history = build_deeplearning_model(X_train, y_train)
        
        #
        # Model evaluation
        #
        print('\nMODEL EVALUATION: \n')
        if USE_RANDOM_FOREST:
            acc_score = evaluate_rf_model(clf, feature_names, label_encoder, X_test, y_test, figsize=(10,10))
            mlflow.log_metric('accuracy', acc_score)
            mlflow.sklearn.log_model(clf, "random_forest_model")
        if USE_XGBOOST:
            acc_score = evaluate_xgboost_model(clf, feature_names, label_encoder, X_test, y_test)
            mlflow.log_metric('accuracy', acc_score)
            mlflow.sklearn.log_model(clf, "xgboost_model")
        if USE_DEEP_LEARNING_WITH_ONEHOT_ENCODING or USE_DEEP_LEARNING_WITH_TARGET_ENCODING:
            test_loss, test_accuracy = evaluate_dl_model(clf, history, label_encoder, X_test, y_test)
            mlflow.log_metric("loss", test_loss)
            mlflow.log_metric("accuracy", test_accuracy)
            mlflow.sklearn.log_model(clf, "deep_learning_model")
                
        print('\n ML FLOW DONE. \n')

    return target_encoders, label_encoder
 
def predict(label_encoder, target_encoders, test_data_file='test_data.csv'):
    '''
    predict results from test dataset in deployment
    '''
    test_data = load_data(test_data_file)
    
    if USE_DEEP_LEARNING_WITH_ONEHOT_ENCODING:
        # Data preprocessing for prediction
        transformed_features = onehot_encoding_preprocessing_for_prediction(test_data)
    else:
        transformed_features = target_encoding_preprocessing_for_prediction(test_data, target_encoders)

    # load the model from disk
    loaded_model = load_model()
    
    # Predict the "Days of Stay" for patients.
    predicted_labels = loaded_model.predict(transformed_features)

    if USE_DEEP_LEARNING_WITH_ONEHOT_ENCODING or USE_DEEP_LEARNING_WITH_TARGET_ENCODING:
        # convert deep learning model (MLP) output vector [a0, a1, ..., a10] to the index with the max value
        predicted_labels = np.apply_along_axis(np.argmax, 1, predicted_labels)

    # convert sequential labeling number 0, 1, ..., 10 to the original label like '21-30'
    predict_range = label_encoder.inverse_transform(predicted_labels)

    # Show original testing data with predicted results.
    result_df = test_data.copy()
    result_df['Stay'] = predict_range
    
    return result_df