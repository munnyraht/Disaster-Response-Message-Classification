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
 * matplotlib
 * sklearn
 * plotly
 
 
 ### Motivation for the project
 This project was engineered by udacity ,
 It is the first project in the Udacity Data Science Nanodegree Program
 Data Analysis is done with the CRISP-DM process.
 
CRISP-DM process is generally used while data mining and is very reliable and user friendly. Here is a short description of the steps involved -
* Understanding the business 
* Understanding the data 
* Preparation of data 
* Modelling 
* Evaluation 
 
 ### Files in the repository 
 
 Project One folder 
 * Dataset folder - contains the Seattle AirBnb datasets
 * analysis.ipynb - Here is the code for the data Analysis
 
### Summary of the results of the analysis
 In summary several factors determines the price of a listing like the month, amenities,cancellation policy and so on. 
 Analysis amd explanation of these are made in the notebook markdowns
 
