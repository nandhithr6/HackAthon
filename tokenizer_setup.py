# tokenizer_setup.py

from transformers import LongformerTokenizerFast
import config # Import configuration

def get_tokenizer(model_name=config.MODEL_NAME):
    """Loads the Longformer tokenizer."""
    try:
        tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        print(f"Tokenizer for '{model_name}' loaded successfully.")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def tokenize_datasets(dataset_dict, tokenizer, text_col=config.TEXT_COLUMN, max_length=config.MAX_LENGTH):
    """Tokenizes the text data in the dataset dictionary."""
    if not tokenizer or not dataset_dict:
        print("Error: Tokenizer or dataset dictionary not provided.")
        return None

    def tokenize_function(examples):
        # Tokenize the text
        return tokenizer(
            examples[text_col],
            padding="max_length", # Pad sequences to max_length
            truncation=True,      # Truncate sequences longer than max_length
            max_length=max_length
        )

    try:
        print(f"Tokenizing datasets with max_length={max_length}...")
        # Ensure the text column exists before removing it
        columns_to_remove = [text_col] if text_col in dataset_dict[list(dataset_dict.keys())[0]].column_names else []
        
        tokenized_datasets = dataset_dict.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove # Remove original text column if it exists
        )

        # Set format for PyTorch
        tokenized_datasets.set_format("torch")
        print("Dataset tokenization complete.")
        return tokenized_datasets
    except Exception as e:
        print(f"An error occurred during tokenization: {e}")
        return None

if __name__ == '__main__':
    # Example Usage (requires data_loader output)
    from data_loader import load_and_split_dataset
    from datasets import DatasetDict # Needed for example usage below

    # 1. Load data
    datasets = load_and_split_dataset()
    if datasets:
        # 2. Get tokenizer
        tokenizer = get_tokenizer()
        if tokenizer:
            # 3. Tokenize data
            tokenized_data = tokenize_datasets(datasets, tokenizer)
            if tokenized_data:
                print("\nTokenized dataset sample (train):")
                # Ensure the 'train' split exists before accessing
                if 'train' in tokenized_data and len(tokenized_data['train']) > 0:
                     print(tokenized_data['train'][0])
                else:
                     print("Train split not found or is empty.")