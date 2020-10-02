# Disaster Response Pipeline Project

### Project description
In the Project Workspace, there is a data set containing real messages that were sent during disaster events. We created a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.
This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data & model evaluation.

### Folder structure:
‘‘‘
- app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- [run.py](http://run.py/) # Flask file that runs app
- data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
- models
|- train_classifier.py
|- classifier.pkl # saved model
- [README.md](http://readme.md/)
’’’


ETL pipeline (process_data.py) extracts, clean and preprocess the data for model training. It also saves the data into sqlite database.

ML pipeline (train_classifier.py) trains a classifier, print out the scores and evaluation details and saves the model into a pickle file & the evaluation data into sqlite database.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

- Command to run ETL pipeline:        
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

- Command to run ML pipeline:
`python models/train_classifier.py data/DisasterResponse.db models/response_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

Credit to Udacity for all provided code templates.
