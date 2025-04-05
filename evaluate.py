# evaluate.py

import torch
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import DatasetDict # Needed for example usage
from data_loader import load_and_split_dataset
from tokenizer_setup import tokenize_datasets # Re-use tokenization logic
from train import compute_metrics # Re-use metrics computation
import config # Import configuration

def evaluate_model(model_path=config.FINAL_MODEL_PATH,
                   test_dataset=None):
    """Evaluates the saved model on the provided test dataset."""

    print(f"\n--- Evaluating Model from {model_path} ---")

    if test_dataset is None:
        print("Error: Test dataset not provided.")
        return None

    try:
        # Load the fine-tuned model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Fine-tuned model and tokenizer loaded successfully.")

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model moved to device: {device}")

        # Minimal TrainingArguments needed for evaluation (includes fp16)
        eval_args = TrainingArguments(
            output_dir="./eval_results", # Temporary directory for evaluation outputs
            per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
            do_train=False,
            do_eval=True,
            report_to="none",
            fp16=torch.cuda.is_available(), # Use mixed precision if available
        )
        print(f"Evaluation arguments configured. FP16 Enabled: {eval_args.fp16}")

        # Initialize Trainer for evaluation
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

        # Perform evaluation
        print("Starting evaluation on the test set...")
        results = trainer.evaluate()
        print("Evaluation complete.")

        # Log and save metrics
        trainer.log_metrics("eval", results)
        trainer.save_metrics("eval", results)

        print("\nTest Set Evaluation Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

        return results

    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Ensure training was successful.")
        return None
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        return None

if __name__ == "__main__":
    # This assumes training has been run and a model saved to config.FINAL_MODEL_PATH

    # 1. Load the original dataset splits
    print("--- Loading Data for Evaluation ---")
    dataset_dict = load_and_split_dataset()
    if not dataset_dict:
        exit()

    # 2. Load tokenizer (essential for processing test data)
    print("\n--- Loading Tokenizer for Evaluation ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.FINAL_MODEL_PATH)
        print(f"Tokenizer loaded from {config.FINAL_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading tokenizer from {config.FINAL_MODEL_PATH}: {e}")
        print("Attempting to load tokenizer using config name...")
        from tokenizer_setup import get_tokenizer # Fallback
        tokenizer = get_tokenizer()
        if not tokenizer:
              print("Failed to load tokenizer. Exiting.")
              exit()

    # 3. Tokenize the test set specifically
    print("\n--- Tokenizing Test Set ---")
    if 'test' not in dataset_dict:
        print("Error: 'test' split not found in dataset dictionary.")
        exit()
        
    # Create a mini-dict just for the test set for tokenization
    test_dict = DatasetDict({'test': dataset_dict['test']})
    tokenized_test_data = tokenize_datasets(test_dict, tokenizer)

    if not tokenized_test_data or 'test' not in tokenized_test_data:
        print("Error: Failed to tokenize the test set.")
        exit()
    else:
        test_dataset_tokenized = tokenized_test_data['test']
        print(f"Test set tokenized with {len(test_dataset_tokenized)} samples.")


    # 4. Evaluate the model
    evaluate_model(test_dataset=test_dataset_tokenized)

    print("\n--- Evaluation Script Finished ---")