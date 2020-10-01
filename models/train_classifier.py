import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tabulate import tabulate
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import FeatureUnion


def load_data(database_filepath):
    """Loads data from db and returns features, labels and label names 
    param: database_filepath: string
    return: X: numpy.ndarray, Y: numpy.ndarray, labels: list
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM TrainingData", con=engine)
    
    labels = [c for c in df.columns if c not in ['id', 'message', 'original', 'genre', 'message_tokn']]
    X = df.message.values
    Y = df[labels].values
    return X, Y, labels


def tokenize(text):
    '''Case normalize, lemmatize, and tokenize the text
    param: text: string
    return: list
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Defines the pipeline and parameter dictionary
       and build a model using GridSearchCV.
    return: RandomForestClassifier
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('multi_clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
#         'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
#         'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
#         'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'multi_clf__estimator': [RandomForestClassifier()],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }
    return GridSearchCV(pipeline, param_grid=parameters)


def evaluate_model(model, X_test, Y_test, category_names, database_filepath):
    """Calculates and prints out the overall accuracy
       and creates a dataframe with accuracy, recall & 
       precission details per label name. This df is also
       printed out and stored into database.
    param: model: A Classifier
    param: X_test,Y_test : numpy.ndarray
    param: category_names: list
    param: database_filepath: string
    """
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean()
    print("Overall Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
    
    df_ytest = pd.DataFrame(data=Y_test, columns=category_names)
    df_ypred = pd.DataFrame(data=y_pred, columns=category_names)

    valid_dict = {'label': [], 'accuracy': [], 'recall': [], 'precision': []}
    for label in category_names:
        valid_dict['label'].append(label)
        valid_dict['accuracy'].append((df_ypred[label] == df_ytest[label]).mean())
        valid_dict['recall'].append(recall_score(df_ytest[label], df_ypred[label], average='weighted'))
        valid_dict['precision'].append(precision_score(df_ytest[label], df_ypred[label], average='weighted'))

    val_df = pd.DataFrame.from_dict(valid_dict)
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    val_df.to_sql('AccuracyDetails', engine, index=False, if_exists='replace')
    print(tabulate(val_df, headers='keys', tablefmt='psql'))
    
  
def save_model(model, model_filepath):
    """Saves the trained model into a pickle file
    param: model: Classifier
    param: model_filepath: String
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

        
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
        evaluate_model(model, X_test, Y_test, category_names, database_filepath)

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