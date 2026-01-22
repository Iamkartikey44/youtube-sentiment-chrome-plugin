import numpy as np
import pandas as pd
import os
import logging
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path,'r') as f:
            params = yaml.safe_load(f)
        logger.debug(f"Parameters retrieved from {params_path}")    

        return params
    except FileNotFoundError:
        logger.error(f"File not found: {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def load_data(data_url:str) -> pd.DataFrame:
    """Load data from CSV file."""

    try:
        df = pd.read_csv(data_url)
        logger.debug(f"Data Loaded from: {data_url}")
        return df
    
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the data: {e}")
        raise

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""

    try:

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['clean_comment'].str.strip() != '']

        logger.debug(f"Data preprocessing completed: Missing values, duplicates, and empty strings removed")
        return df
    except KeyError as e:
        logger.error(f'Missing column in the dataframe: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error during preprocessing: {e}')
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        
        # Create the data/raw directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug(f'Train and test data saved to {raw_data_path}')
    except Exception as e:
        logger.error(f'Unexpected error occurred while saving the data: {e}')
        raise  

def main():

    try:

        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../params.yaml'))
        test_size = params['data_ingestion']['test_size']

        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

        final_df = preprocess_data(df)
        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=42)

        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))

    except Exception as e:
        logger.error(f"Failed to complete the data ingestion process: {e}")
        print(f"Error: {e}") 



if __name__ == "__main__":
    main()