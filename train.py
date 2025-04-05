# train.py

import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, recall_score
import config # Import configuration
from data_loader import load_and_split_dataset
from tokenizer_setup import get_tokenizer, tokenize_datasets
from model import setup_model

# Check GPU availability early
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB") # Display VRAM

# Updated metrics function
def compute_metrics(eval_pred):
    """Computes accuracy, F1, and recall for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(labels, predictions, average='binary', zero_division=0)
    return {
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1,
    }

def main():
    """Main function to orchestrate the training process."""
    print("--- Starting Training Pipeline ---")

    # 1. Load and Split Data
    print("\n--- Loading Data ---")
    dataset_dict = load_and_split_dataset()
    if not dataset_dict:
        return

    # 2. Setup Tokenizer
    print("\n--- Setting up Tokenizer ---")
    tokenizer = get_tokenizer()
    if not tokenizer:
        return

    # 3. Tokenize Data
    print("\n--- Tokenizing Data ---")
    tokenized_datasets = tokenize_datasets(dataset_dict, tokenizer)
    if not tokenized_datasets:
        return

    # 4. Setup Model
    print("\n--- Setting up Model ---")
    model, device = setup_model() # setup_model already moves the model to the correct device
    if not model:
        return

    # 5. Setup Training Arguments (Includes gradient accumulation and fp16)
    print("\n--- Configuring Training Arguments ---")
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        logging_dir=config.LOGGING_DIR,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS, # For GPU memory optimization
        fp16=torch.cuda.is_available(),                               # Use FP16 if GPU is available
        evaluation_strategy=config.EVALUATION_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        report_to=config.REPORT_TO,
        logging_steps=config.LOGGING_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        # optim="adamw_torch_fused", # Consider if using PyTorch 2.0+
    )
    print(f"Training arguments configured. Output directory: {config.OUTPUT_DIR}")
    print(f"  FP16 Enabled: {training_args.fp16}")
    print(f"  Train Batch Size (per device): {training_args.per_device_train_batch_size}")
    print(f"  Gradient Accumulation Steps: {training_args.gradient_accumulation_steps}")
    # Calculate effective batch size considering potential multiple GPUs in future
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_gpus
    print(f"  Effective Batch Size: {effective_batch_size}")


    # 6. Initialize Trainer
    print("\n--- Initializing Trainer ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer, # Pass tokenizer for easy saving
    )
    print("Trainer initialized.")

    # 7. Start Training
    print("\n--- Starting Training ---")
    try:
        train_result = trainer.train()
        print("Training finished.")
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # 8. Save the best model
        print("\n--- Saving Final Model ---")
        trainer.save_model(config.FINAL_MODEL_PATH)
        print(f"Best model saved to {config.FINAL_MODEL_PATH}")

    except torch.cuda.OutOfMemoryError: # Catch specific OOM errors
        print("\n--- ERROR: CUDA Out of Memory ---")
        print("Try reducing 'PER_DEVICE_TRAIN_BATCH_SIZE' or 'MAX_LENGTH' in config.py,")
        print("or increasing 'GRADIENT_ACCUMULATION_STEPS'.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")

    print("\n--- Training Pipeline Complete ---")

if __name__ == "__main__":
    main()