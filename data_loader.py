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
    """Loads the dataset from CSV, ensures integer labels, uses a subset, and splits it."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded from {csv_path} with {len(df)} samples")

        # Ensure correct column names
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns.")

        # --- Ensure the label column is integer type ---
        try:
            df[label_col] = df[label_col].astype(int)
            print(f"Converted '{label_col}' column to integer type.")
        except ValueError as e:
            print(f"Error converting label column '{label_col}' to integer: {e}")
            print("Please ensure the label column contains only values convertible to 0 or 1.")
            return None
        # --- End label conversion fix ---

        # Rename label column if necessary to 'labels' for Hugging Face compatibility
        if label_col != 'labels':
            df = df.rename(columns={label_col: 'labels'})
            label_col_hf = 'labels'
        else:
            label_col_hf = label_col

        print(f"Sample data (after potential label conversion):\n{df.head()}")

        # Convert to HuggingFace dataset format
        dataset = Dataset.from_pandas(df[[text_col, label_col_hf]])

        # --- START SUBSET SELECTION ---
        # Keep only a small fraction for faster training during hackathon
        subset_size = 10000 # Adjust this number based on time! Start with 10k or 5k.
        total_samples_full = len(dataset)
        if subset_size < total_samples_full:
             # Shuffle before selecting might be good if data is ordered
             # dataset = dataset.shuffle(seed=seed)
             dataset = dataset.select(range(subset_size))
             print(f"--- Using a subset of {subset_size} samples for training ---")
        else:
             print("--- Using full dataset (subset size >= total samples) ---")
        # --- END SUBSET SELECTION ---

        # Split the potentially smaller dataset
        train_testval = dataset.train_test_split(test_size=test_size, seed=seed)
        # Split the test set further into validation and test
        test_val = train_testval['test'].train_test_split(test_size=validation_size, seed=seed)

        dataset_dict = DatasetDict({
            'train': train_testval['train'],
            'validation': test_val['train'],
            'test': test_val['test']
        })

        # Print split sizes (based on the subset)
        total_samples_subset = len(dataset)
        print(f"Train set (subset): {len(dataset_dict['train'])} samples ({len(dataset_dict['train'])/total_samples_subset:.1%})")
        print(f"Validation set (subset): {len(dataset_dict['validation'])} samples ({len(dataset_dict['validation'])/total_samples_subset:.1%})")
        print(f"Test set (subset): {len(dataset_dict['test'])} samples ({len(dataset_dict['test'])/total_samples_subset:.1%})")

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
        if 'train' in datasets and len(datasets['train']) > 0:
             print("\nSample label from train dataset:", datasets['train'][0]['labels'])
             print("Data type of labels:", type(datasets['train'][0]['labels']))
             print("Dataset features:", datasets['train'].features)