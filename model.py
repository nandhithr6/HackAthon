# model.py

import torch
from transformers import LongformerForSequenceClassification
import config # Import configuration

def setup_model(model_name=config.MODEL_NAME,
                num_labels=config.NUM_LABELS,
                layers_to_freeze=config.LAYERS_TO_FREEZE):
    """Loads the Longformer model and optionally freezes initial layers."""
    try:
        # Load the model
        model = LongformerForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        print(f"Model '{model_name}' loaded successfully for {num_labels}-class classification.")

        # Freeze specified layers
        if layers_to_freeze > 0:
            print(f"Freezing the first {layers_to_freeze} encoder layers...")
            # Ensure the encoder structure exists before accessing layers
            if hasattr(model, 'longformer') and hasattr(model.longformer, 'encoder') and hasattr(model.longformer.encoder, 'layer'):
                for i, layer in enumerate(model.longformer.encoder.layer):
                    if i < layers_to_freeze:
                        for param in layer.parameters():
                            param.requires_grad = False
                        # print(f"Layer {i} frozen.") # Uncomment for detailed logging
                    else:
                         # Ensure subsequent layers are trainable (might be default, but good practice)
                         for param in layer.parameters():
                             param.requires_grad = True
                print(f"Finished freezing {layers_to_freeze} layers.")
            else:
                print("Warning: Could not find expected encoder structure to freeze layers.")
        else:
            print("No layers frozen. Training all layers.")

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model moved to device: {device}")

        return model, device

    except Exception as e:
        print(f"Error setting up the model: {e}")
        return None, None

if __name__ == '__main__':
    # Example usage
    model, device = setup_model()
    if model:
        print("\nModel setup complete.")
        # You can print model structure or check trainable parameters here
        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f"Total parameters: {total_params}")
        # print(f"Trainable parameters: {trainable_params}")