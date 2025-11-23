import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Logging configuration
logger = logging.getLogger('Data preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/preprocessing.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NTLK data
nltk.download('stopwords')
nltk.download('wordnet')


# Define preprocess function
def preprocess_comment(comment: str) -> str:
    """"Apply preprocessing transformations to a comment."""
    try:
        # Convert to lower case
        comment = comment.lower()

        # Remove trailing and leading whitespace
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopword but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - \
            {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join(
            [word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word)
                           for word in comment.split()])

        return comment

    except Exception as e:
        logger.error(f"Unexpected error when preprocessing comment: {e}")
        raise


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the text data in the dataframe"""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug(f"Text normalization completed.")
        return df

    except Exception as e:
        logger.error(f"Unexpected error when normalizing text: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """"Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')

        # Create the path data/interim directory if does not exist
        os.makedirs(interim_data_path, exist_ok=True)

        # Save the train and test data
        train_data.to_csv(os.path.join(
            interim_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(
            interim_data_path, "test.csv"), index=False)

        logger.debug(f"Train and test data saved to: {interim_data_path}")

    except Exception as e:
        logger.error(f"Unexpected error when saving datasets: {e}")
        raise


def main():
    try:
        logger.debug("Starting data preprocessing ...")

        # Fetch data from  data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        # Preprocess the data
        preprocess_train_data = normalize_text(df=train_data)
        preprocess_test_data = normalize_text(df=test_data)

        # Saving preprocessed data
        save_data(train_data=preprocess_train_data,
                  test_data=preprocess_test_data, data_path=('./data'))
    except Exception as e:
        logger.error(f"Failed to preprocessing data: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
