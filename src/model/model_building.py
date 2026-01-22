import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters retrieved from: {params_path} ")
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

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug(f"Data loaded and NaNs filled from: {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the data: {e}")
        raise

def apply_tfidf(train_data: pd.DataFrame, max_features:int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF with ngrams to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        #Perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(f"TF-IDF transformation complete. Train Shape: {X_train_tfidf.shape}")

        #Save the vectorizer in the root directory
        with open(os.path.join(get_root_directory(),'tfidf_vectorizer.pkl'),'wb') as f:
            pickle.dump(vectorizer,f)
        logger.debug(f"TF-IDF applied with trigrams and data transformed")

        return X_train_tfidf,y_train
    except Exception as e:
        logger.error(f"Error during TF-IDF transformation: {e}")
        raise

def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int,n_estimators: int) -> lgb.LGBMClassifier:
    """Train a LightGBM model."""
    try:
        best_model = lgb.LGBMClassifier(objective ='multiclass',num_class=3,metric='multi_logloss',is_unbalance=True,class_weight="balanced",reg_alpha=0.1,
                                        reg_lambda=0.1,learning_rate=learning_rate,max_depth=max_depth,n_estimators=n_estimators)
        best_model.fit(X_train,y_train)
        logger.debug("LightGBM model training completed.")
        return best_model
    except Exception as e:
        logger.error(f"Error during LightGBM model training: {e}")
        raise

def save_model(model,file_path: str)-> None:
    """Save the trained model to a file."""
    try:
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug(f"Model saved to : {file_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving the model: {e}")
        raise

def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir,'params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        train_data = load_data(os.path.join(root_dir,'data/interim/train_processed.csv'))

        X_train_tfidf,y_train = apply_tfidf(train_data,max_features,ngram_range)
        best_model = train_lgbm(X_train_tfidf,y_train,learning_rate,max_depth,n_estimators)

        save_model(best_model,os.path.join(root_dir,'lgbm_model.pkl'))

    except Exception as e:
        logger.error(f"Failed to complete the feature engineering and model building process: {e}")
        raise



if __name__ == '__main__':
    main()         
