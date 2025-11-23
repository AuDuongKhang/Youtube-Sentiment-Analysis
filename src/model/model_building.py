import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer


# Logging configuration
logger = logging.getLogger('Model building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/model_building.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """"Load parameters from a YAML file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters retrieve from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"File not found {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when loading parameters: {e}")
        raise
    
    
def load_data(data_path: str) -> pd.DataFrame:
    """"Load data from a CSV file"""
    try:
        df = pd.read_csv(data_path)
        df.fillna('', inplace=True) # Fill any NaN values
        logger.debug(f"Data loaded and NaN filled from: {data_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse to CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when loading data: {e}")
        raise
    

def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """"Apply TF-IDF with ngrams to train data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        
        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values
        
        # Perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        logger.debug(f"TF-IDF transformation completed. Train data shape: {X_train_tfidf.shape}")
        
        # Save the vectorizer in the model path
        os.makedirs('model', exist_ok=True)
        with open('./model/tfidf_vectorizer.pkl', 'wb') as file: 
            pickle.dump(vectorizer, file)
            
        logger.debug("TF-IDF applied with ngrams and data transformed.")
        
        return X_train_tfidf, y_train
    
    except Exception as e: 
        logger.error(f"Unexpected error when apply TF-IDF: {e}")
        raise
    

def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """"Train a LightGBM model"""
    try:
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            is_unbalance=True,
            class_weight='balanced',
            reg_alpha=0.1, # L1 regularization
            reg_lambda=0.1, # L2 regularization
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )        
        model.fit(X_train, y_train)
        logger.debug("LightGBM model training completed.")
        
        return model
    
    except Exception as e: 
        logger.error(f"Unexpected error when training model: {e}")
        raise
    
    
def save_model(model, file_path: str) -> None:
    """"Save the trained model to a file"""
    try:
        with open(file_path, 'wb') as file: 
            pickle.dump(model, file)
        logger.debug(f"Model saved to: {file}")
    except Exception as e: 
        logger.error(f"Unexpected error when saving model: {e}")
        raise
    

def main():
    try:
        # Load parameters from the params.yaml in the config directory
        params = load_params(params_path='./config/params.yaml')
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])
              
        # Fetch data from  data/interim
        train_data = load_data('./data/interim/train.csv')
        
        # Apply TF-IDF
        X_train_tfidf, y_train = apply_tfidf(train_data=train_data, max_features=max_features, ngram_range=ngram_range)
        
        # Load parameters from the params.yaml in the config directory
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']
        # Train model
        model = train_lgbm(X_train=X_train_tfidf, y_train=y_train,learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
        
        # Save model
        os.makedirs('model', exist_ok=True)
        save_model(model, './model/lgbm_model.pkl')
        
    except Exception as e: 
        logger.error(f"Failed to build model: {e}")
        print(f"Error: {e}")
        

if __name__ == '__main__':
    main()