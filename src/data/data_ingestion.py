import numpy as np
import pandas as pd
import yaml
import logging
import os

from sklearn.model_selection import train_test_split

# Logging configuration
logger = logging.getLogger('Data ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/error.log')
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


def load_data(data_url: str) -> pd.DataFrame:
    """"Load data from a CSV file"""
    try:
        df = pd.read_csv(data_url)
        logger.debug(f"Data loaded from: {data_url}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {data_url}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse to CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when loading data: {e}")
        raise


def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:
    """"Preprocessing data by handling missing value, duplicates, and empty strings."""
    try:
        # Remove missing value
        df.dropna(inplace=True)
        # Remove duplicate
        df.drop_duplicates(inplace=True)
        # Remove rows with empty strings
        df = df[df['clean_comment'].str.strip() != '']

        logger.debug(
            "Data preprocessing completed: Missing value, duplicates, and empty string removed.")
        return df
    except KeyError as e:
        logger.error(f"Missing column in dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when preprocessing: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """"Save the train and test datasets, creating a raw folder if it does not exist"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')

        # Create the path data/raw directory if does not exist
        os.makedirs(raw_data_path, exist_ok=True)

        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug(f"Train and test data saved to: {raw_data_path}")

    except Exception as e:
        logger.error(f"Unexpected error when saving datasets: {e}")
        raise


def main():
    try:
        # Load parameters from the params.yaml in the root directory
        params = load_params(params_path='./config/params.yaml')

        test_size = params['data_ingestion']['test_size']

        # Load data from specified URL
        df = load_data(
            data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

        # Preprocessing data
        final_df = preprocessing_data(df=df)

        # Split data into training and testing data
        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42)

        os.makedirs('./data', exist_ok=True)
        # Save datasets
        save_data(train_data=train_data,
                  test_data=test_data, data_path='./data')

    except Exception as e:
        logger.error(f"Failed to complete the data ingestion process: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
