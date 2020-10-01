import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''Case normalize, lemmatize, and tokenize the text'''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def top_used_labels(df):
    """Extracts the top 10 labels by count of related messages
       Returns: pandas df   
    """
    labels = [c for c in df.columns if c not in ['id', 'message', 'original', 'genre', 'message_tokn', 'related']]
    label_dict = {'label': [], 'count': []}
    for label in labels:
        label_dict['count'].append(df[label].sum())
        label_dict['label'].append(label)

    ldf = pd.DataFrame.from_dict(label_dict)
    return ldf.sort_values(by=['count'], ascending=False).head(10)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('TrainingData', engine)
val_df = pd.read_sql_table('AccuracyDetails', engine)
ldf = top_used_labels(df)

# load model
model = joblib.load("../models/response_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
#     genre_counts = df.groupby('genre').count()['message']
#     genre_names = list(genre_counts.index)
    accuracy = val_df.sort_values(by=['accuracy'])['accuracy'].tolist()
    label_names = val_df.sort_values(by=['accuracy'])['label'].tolist()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=label_names,
                    y=accuracy
                )
            ],

            'layout': {
                'title': 'Accuracy by label',
                'yaxis': {
                    'title': "Accuracy"
                },
                'xaxis': {
                    'title': "Label name"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=ldf['label'].tolist(),
                    y=ldf['count'].tolist()
                )
            ],

            'layout': {
                'title': 'Top 10 used categories',
                'yaxis': {
                    'title': "Sum of related messages"
                },
                'xaxis': {
                    'title': "Label name"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    print(query)
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()