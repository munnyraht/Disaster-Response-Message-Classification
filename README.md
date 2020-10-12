# Disaster Response Message Classification
### Disaster Response Pipeline Project
A Flask app that classifies a disaster response message into 36 categories

### Instructions:
First, download the data set from : https://drive.google.com/drive/folders/1HSbVyyXSXWuEHHogFxyFYx6SxngO6qmE?usp=sharing
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



# ETL Pipeline
 * Loading data from csv files
 * Prepare Clean data, re arrange categories.
 * Save data into sqlite data base
 
# ML Pipeline
 * Load data from database
 * Build model with multioutput multiclassification estimator (MultiOutputClassifier and KNeighborsClassifier), and use Gridsearch cv to get the best parameter for    the model
 * Training model 
 * Evaluating model - calculate the recall score, precision score and F1 score of the model for each categories
 * Saving model - save trained model into a pickle file

 
 ## #Libraries used:
 * pandas
 * numpy
 * scipy
 * SQLAlchemy
 * nltk
 * sklearn
 * plotly
 
 
 ### Motivation for the project
 This project was engineered by udacity ,
 This aim of this project is classify a disaster response message into different categories in other to recognise which response unit to be alerted and to response   to the messages.

 ### Files in the repository 
 
 * Disaster-response 
  ** template
    *** go.html - an htlml file with jinja code to display the classification of the message
    *** master.html - an html file with base codes that extends its templates
  ** run.py - a script that loads the home page and routes to classify messages with the classifier
 * data
    ** process_data.py - a script that cleans data and save data in db,
    ** DisasterResponse.db - a result from the process_data.py
    ** disaster_messages.csv - data to be processed
    ** disaster_categories.csv - data to be processed
 * models
    ** train_classifier.py - read in data, tokenize, build a model and save the model as a pickle file
 * notebooks
   ** ETL Pipeline Preparation.ipynb - step by step on how to process , read and save to db
   ** ML Pipeline Preparatio.ipynb - step by step on how to build a model and improve it with GridSearchCv
 
