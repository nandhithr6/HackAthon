# data_loader.py

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import config # Import configuration

def load_and_split_dataset(csv_path=config.DATA_PATH,
                           text_col=config.TEXT_COLUMN,
                           label_col=config.LABEL_COLUMN,
                           test_size=config.TEST_SPLIT_SIZE,
                           validation_size=config.VALIDATION_SPLIT_SIZE,
                           seed=config.RANDOM_SEED):
    """Loads the dataset from CSV and splits it into train, validation, and test sets."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded from {csv_path} with {len(df)} samples")

        # Ensure correct column names
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns.")

        # Rename label column if necessary to 'labels' for Hugging Face compatibility
        if label_col != 'labels':
            df = df.rename(columns={label_col: 'labels'})

        print(f"Sample data:\n{df.head()}")

        # Convert to HuggingFace dataset format
        dataset = Dataset.from_pandas(df[[text_col, 'labels']]) # Keep only necessary columns

        # Split the dataset
        train_testval = dataset.train_test_split(test_size=test_size, seed=seed)
        # Split the test set further into validation and test
        test_val = train_testval['test'].train_test_split(test_size=validation_size, seed=seed)

        dataset_dict = DatasetDict({
            'train': train_testval['train'],
            'validation': test_val['train'],
            'test': test_val['test']
        })

        # Print split sizes
        total_samples = len(dataset)
        print(f"Train set: {len(dataset_dict['train'])} samples ({len(dataset_dict['train'])/total_samples:.1%})")
        print(f"Validation set: {len(dataset_dict['validation'])} samples ({len(dataset_dict['validation'])/total_samples:.1%})")
        print(f"Test set: {len(dataset_dict['test'])} samples ({len(dataset_dict['test'])/total_samples:.1%})")

        return dataset_dict

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {csv_path}")
        return None
    except ValueError as ve:
        print(f"Error: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None

if __name__ == '__main__':
    # Example usage: Load and split the data
    datasets = load_and_split_dataset()
    if datasets:
        print("\nDataset splits created successfully:")
        print(datasets)