import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,recall_score,f1_score
from sklearn.preprocessing import label_binarize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    conn = engine.connect()
    df = pd.read_sql_table('DisasterResponse',conn)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()
    return X,Y,category_names


def tokenize(text):
    tokenized = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    clean_token = []
    for token in tokenized:
        clean_token.append(lemmatizer.lemmatize(token))
    return clean_token
 

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
         ])                                
                          
    params={'clf__estimator__n_neighbors':(4, 5)
        }

    cv = GridSearchCV(pipeline, param_grid=params)
     
    return cv 


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    for i,column in enumerate(Y_test.columns):
        print('report for {}'.format(column))
        print('recall score:',recall_score(Y_test[column].values.astype(str),y_pred[:,i].astype(int).astype(str),average='micro'))
        print('precision_score:',precision_score(Y_test[column].values.astype(int),y_pred[:,i].astype(int),average='micro'))
        print('f1 score:',f1_score(Y_test[column].values.astype(int),y_pred[:,i].astype(int),average='micro'), '\n')
    return


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    return

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
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