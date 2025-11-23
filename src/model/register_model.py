import json
import mlflow
import logging
import os

# Setup MLflow tracking uri
mlflow.set_tracking_uri('http://ec2-16-176-5-94.ap-southeast-2.compute.amazonaws.com:5000')

# Logging configuration
logger = logging.getLogger('Register model')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/register_model.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """"Load the model information from JSON file"""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        
        logger.debug(f"Model info loaded from: {file_path}")
        return model_info
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when loading model information: {e}")
        raise
    

def register_model(model_name: str, model_info: dict) -> None:
    """"Register the model to the MLflow Model Registry"""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Staging'
        )
        
        logger.debug(f"Model {model_name} version {model_version} registered and transitioned to Staging")
    except Exception as e: 
        logger.error(f"Unexpected error when register model: {e}")
        raise
    

def main():
    try:
        model_info_path = './experiments/experiment_info.json'
        # Load model information
        model_info = load_model_info(file_path=model_info_path)
        
        model_name = 'yt_chrome_plugin_model'
        # Register model
        register_model(model_name=model_name, model_info=model_info)
        logger.debug("Register model completed.")
        
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        print(f"Error: {e}")
    
    
if __name__ == '__main__':
    main()
        