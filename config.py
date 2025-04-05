# config.py

# Model configuration
MODEL_NAME = "allenai/longformer-base-4096"
NUM_LABELS = 2
MAX_LENGTH = 1024 # Tokenizer max length (adjust if needed, but 1024 is safer for 8GB VRAM)
LAYERS_TO_FREEZE = 6

# Data configuration
DATA_PATH = "ai_vs_human.csv" # Path to your dataset CSV
TEXT_COLUMN = "text"
LABEL_COLUMN = "label" # Assuming your label column is named 'label'
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.5 # Proportion of the initial test split to use for validation (0.5 means 10% of total data)
RANDOM_SEED = 42

# Training configuration
OUTPUT_DIR = "./results"
LOGGING_DIR = "./logs"
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
# Keep batch size low for 8GB VRAM with Longformer; use gradient accumulation
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1 # Evaluation might handle slightly larger, but 1 is safe
# NEW: Gradient Accumulation to increase effective batch size
# Effective batch size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * num_gpus
# E.g., 1 * 8 * 1 = effective batch size of 8
GRADIENT_ACCUMULATION_STEPS = 8 # Adjust as needed (4, 8, 16 are common)

EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1" # Use F1 score to select the best model
REPORT_TO = "none" # Or "tensorboard", "wandb"
LOGGING_STEPS = 50 # How often to log training loss
SAVE_TOTAL_LIMIT = 2 # Keep only the best and the latest checkpoint

# Prediction configuration
SCORE_THRESHOLD = 50 # Example threshold (0-100 scale)

# Saved model path
FINAL_MODEL_PATH = "./final_model"