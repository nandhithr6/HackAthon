# predict.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax, sigmoid
import config # Import configuration

# Load model and tokenizer globally or within a class for efficiency if predicting many times
# Wrapped in a function to handle potential loading errors gracefully
def load_prediction_model_tokenizer(model_path=config.FINAL_MODEL_PATH):
    """Loads the model and tokenizer for prediction."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() # Set model to evaluation mode
        print(f"Model and tokenizer loaded from {model_path} onto {device}.")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model/tokenizer from {model_path}: {e}")
        return None, None, None

# Load globally on script start
predictor_model, predictor_tokenizer, predictor_device = load_prediction_model_tokenizer()

def predict_text(text, threshold=config.SCORE_THRESHOLD):
    """
    Predicts if the text is AI-generated or human-written and provides a score.

    Args:
        text (str): The input text to classify.
        threshold (int): The threshold (0-100) for classifying as AI-generated.

    Returns:
        dict: A dictionary containing the prediction, score, and label.
              Returns None if the model/tokenizer failed to load.
    """
    if not predictor_model or not predictor_tokenizer:
        print("Model or tokenizer not loaded. Cannot predict.")
        return None

    try:
        # 1. Tokenize the input text
        inputs = predictor_tokenizer(text,
                           return_tensors="pt", # Return PyTorch tensors
                           max_length=config.MAX_LENGTH,
                           padding="max_length",
                           truncation=True)

        # Move tensors to the correct device
        inputs = {k: v.to(predictor_device) for k, v in inputs.items()}

        # 2. Get model predictions (no gradient calculation needed)
        with torch.no_grad():
            outputs = predictor_model(**inputs)
            logits = outputs.logits # Raw scores from the model

        # 3. Calculate probability/score using Sigmoid on the 'AI' logit (index 1)
        # Ensure index 1 corresponds to the AI class based on your training data/labels
        ai_logit = logits[0, 1]
        ai_prob = sigmoid(ai_logit).item()

        # Convert probability (0-1) to score (0-100)
        score = ai_prob * 100

        # 4. Classify based on threshold
        predicted_label_index = 1 if score >= threshold else 0
        predicted_label = "AI-Generated" if predicted_label_index == 1 else "Human-Written"

        return {
            "predicted_label": predicted_label,
            "score": score, # Score (0-100) indicating likelihood of being AI-generated
            "raw_logits": logits.squeeze().tolist() # Optional: include raw logits
        }

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

if __name__ == "__main__":
    # Example Usage
    if predictor_model and predictor_tokenizer: # Only run examples if model loaded
        test_text_human = "I walked my dog in the park this morning. It was lovely and sunny, and he chased a squirrel."
        test_text_ai = "The synergistic application of blockchain technology and artificial intelligence facilitates unprecedented levels of data integrity and automated decision-making processes across decentralized networks."

        print("\n--- Prediction Examples ---")

        prediction1 = predict_text(test_text_human)
        if prediction1:
            print(f"\nInput Text: '{test_text_human}'")
            print(f"Prediction: {prediction1['predicted_label']}")
            print(f"AI Score: {prediction1['score']:.2f}")

        prediction2 = predict_text(test_text_ai)
        if prediction2:
            print(f"\nInput Text: '{test_text_ai}'")
            print(f"Prediction: {prediction2['predicted_label']}")
            print(f"AI Score: {prediction2['score']:.2f}")

        # Example with custom threshold
        prediction3 = predict_text(test_text_ai, threshold=75)
        if prediction3:
            print(f"\nInput Text (Threshold 75): '{test_text_ai}'")
            print(f"Prediction: {prediction3['predicted_label']}")
            print(f"AI Score: {prediction3['score']:.2f}")
    else:
        print("\nSkipping prediction examples because the model/tokenizer could not be loaded.")