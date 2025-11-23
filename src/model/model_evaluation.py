import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# Logging configuration
logger = logging.getLogger('Model evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/model_evaluation.log')
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
    
    
def load_model(model_path: str):
    """"Load trained model"""
    try:
        with open(model_path, 'rb') as file: 
            model = pickle.load(file)
        logger.debug(f"Model loaded from: {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"File not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when loading model: {e}")
        raise
    

def load_vectorizer(model_path: str):
    """"Load vectorizer model"""
    try:
        with open(model_path, 'rb') as file: 
            vectorizer = pickle.load(file)
        logger.debug(f"TF-IDF vectorizer model loaded from: {model_path}")
        return vectorizer
    except FileNotFoundError:
        logger.error(f"File not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when loading vectorizer model: {e}")
        raise
    

def evaluation_model(model, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """"Evaluate model, log classification metrics and confusion matrix"""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug("Evaluation completed.")
        
        return report, cm
    except Exception as e: 
        logger.error(f"Unexpected error when evaluating: {e}")
        raise    


def log_confusion_matrix(cm, dataset_name: str, file_path) -> None:
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = os.path.join(file_path, f'confusion_matrix_{dataset_name}.png')
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()
    
    
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """"Save the model run ID and path to a JSON file"""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        
        # Save a dictionary as JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug(f"Model information saved to: {file_path}")
    except Exception as e: 
        logger.error(f"Unexpected error when saving model information: {e}")
        raise
    
def main():
    mlflow.set_tracking_uri('http://ec2-16-176-5-94.ap-southeast-2.compute.amazonaws.com:5000')
    
    mlflow.set_experiment('dvc-pipeline-runs')
    
    with mlflow.start_run() as run:
        try:
            # Load parameters from the params.yaml in the config directory
            params = load_params(params_path='./config/params.yaml')
            
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
                
            # Load model and vectorizer
            model = load_model(model_path='./model/lgbm_model.pkl')
            vectorizer = load_vectorizer(model_path='./model/tfidf_vectorizer.pkl')
            
            # Load test data
            test_data = load_data(data_path='./data/interim/test.csv')
            
            # Prepare data
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values
            
            # Create data frames for signature inference (using for first few rows as an example)
            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())
            
            # Infer the signature
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5])) # Added for signature
            
            # Log model with signature
            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                input_example=input_example # Added input example
            )

            # Save model info
            os.makedirs('experiments', exist_ok=True)
            model_path = 'lgbm_model'
            save_model_info(run.info.run_id, model_path=model_path, file_path='./experiments/experiment_info.json')
            
            # Log the vectorizer as an artifact
            mlflow.log_artifact('./model/tfidf_vectorizer.pkl')
            
            # Evaluate model and get metrics
            report, cm = evaluation_model(model=model, X_test=X_test_tfidf, y_test=y_test)
            
            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })
            
            os.makedirs('./training_result/', exist_ok=True)
            log_confusion_matrix(cm=cm,dataset_name="Test data", file_path='./training_result/')
            
            # Add important tags
            mlflow.set_tag('model_type', 'LightGBM')
            mlflow.set_tag('task', 'Sentiment Analysis')
            mlflow.set_tag('dataset', 'YouTube Comment')
            
            logger.debug('Evaluation completed.')
            
        except Exception as e: 
            logger.error(f"Failed to evaluate model: {e}")
            print(f"Error: {e}")    
    
if __name__ == "__main__":
    main()
    
