import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
      INPUT - messages_file path: path to the messages csv file
              categories_filepath: path to the categories csv file   

      OUTPUT - df: a merged dataframe of messages and categories data 
      '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on= 'id')
    return df


def clean_data(df):
    '''
      INPUT - df : dataframe to be cleaned
      OUTPUT - df: cleaned df 
      '''
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True))
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # extract category name for row
    category_colnames = row.apply(lambda x:x[:-2]).tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(str)
        
    # drop the original categories column from `df`
    df.drop(['categories'],inplace=True, axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
      INPUT - df : dataframe to be saved to db
      OUTPUT - database_filename 
      '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename[5:-3], engine, index=False, if_exists='replace')
    print(database_filename[:-3])
    return  


def main():
    '''
      A function that loads dataset, cleans data, and saves to db
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