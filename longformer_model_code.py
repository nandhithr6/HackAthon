# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 📁 Dataset Loading
def load_and_split_dataset(csv_path="ai_vs_human.csv"):
    # Load the dataset
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded with {len(df)} samples")
    print(f"Sample data:\n{df.head()}")
    
    # Convert to HuggingFace dataset format
    dataset = Dataset.from_pandas(df)
    
    # Split the dataset into train, validation, and test sets
    train_testval = dataset.train_test_split(test_size=0.2, seed=42)
    test_val = train_testval['test'].train_test_split(test_size=0.5, seed=42)
    
    # Create the final dataset dictionary
    dataset_dict = DatasetDict({
        'train': train_testval['train'],
        'validation': test_val['train'],
        'test': test_val['test']
    })
    
    # Print split sizes
    print(f"Train set: {len(dataset_dict['train'])} samples ({len(dataset_dict['train'])/len(dataset):.1%})")
    print(f"Validation set: {len(dataset_dict['validation'])} samples ({len(dataset_dict['validation'])/len(dataset):.1%})")
    print(f"Test set: {len(dataset_dict['test'])} samples ({len(dataset_dict['test'])/len(dataset):.1%})")
    
    return dataset_dict

# 🔠 Tokenizer Setup
def setup_tokenizer():
    tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
    print("Tokenizer loaded successfully")
    return tokenizer

# Tokenize the dataset
def tokenize_dataset(dataset_dict, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=1024
        )
    
    # Apply tokenization to all splits
    tokenized_datasets = dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Set the format for PyTorch
    tokenized_datasets.set_format("torch")
    print("Dataset tokenized successfully")
    
    return tokenized_datasets

# 🧠 Model Setup
def setup_model():
    # Load the Longformer model for sequence classification
    model = LongformerForSequenceClassification.from_pretrained(
        "allenai/longformer-base-4096",
        num_labels=2
    )
    
    # 🧊 Freeze the first 6 layers
    for i, layer in enumerate(model.longformer.encoder.layer):
        if i < 6:
            for param in layer.parameters():
                param.requires_grad = False
            print(f"Layer {i} frozen")
    
    # Move model to GPU
    model.to(device)
    print("Model loaded and configured successfully")
    
    return model

# 📊 Metrics Setup
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {"accuracy": accuracy, "f1": f1}

# ⚙️ Training Setup
def setup_training(model, tokenized_datasets):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Changeable based on GPU memory
        per_device_eval_batch_size=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        report_to="none"
    )
    
    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )
    
    print("Trainer setup complete")
    return trainer

# Main execution flow
def main():
    # Load and split the dataset
    dataset_dict = load_and_split_dataset()
    
    # Setup tokenizer
    tokenizer = setup_tokenizer()
    
    # Tokenize dataset
    tokenized_datasets = tokenize_dataset(dataset_dict, tokenizer)
    
    # Setup model
    model = setup_model()
    
    # Setup training
    trainer = setup_training(model, tokenized_datasets)
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test results: {test_results}")
    
    # Save the model
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("Model and tokenizer saved successfully")

# Uncomment to run the main function
# if __name__ == "__main__":
#     main()
