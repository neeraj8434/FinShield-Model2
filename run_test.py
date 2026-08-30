import torch
from src.model2_reference import Model

def main():
    print("Initializing Model...")
    # Initialize the model with 2 classes (e.g., Fake vs Real)
    # The use_frequency_feature flag is True by default
    model = Model(num_classes=2, use_frequency_feature=True)
    
    # Put the model in evaluation mode
    model.eval()

    print("Model initialized successfully!")
    print("Creating mock video batch...")
    
    # Create mock video input data
    # Shape: (batch_size, seq_length, channels, height, width)
    # Let's assume a batch of 2 videos, 5 frames each, 3 channels (RGB), 224x224 pixels
    batch_size = 2
    seq_length = 5
    c, h, w = 3, 224, 224
    
    # Generate random tensor with values between 0 and 1
    mock_input = torch.rand(batch_size, seq_length, c, h, w)
    
    print(f"Mock input shape: {mock_input.shape}")
    print("Running forward pass (this might take a few seconds)...")
    
    # Run the model forward pass (default: use_fusion_classifier=True → 4-tuple)
    with torch.no_grad():
        fmap, logits, freq_score, temporal_score = model(mock_input)
        
    print("\n--- Output Results ---")
    print(f"Feature Map shape: {fmap.shape}")
    print(f"Logits (Fused classification output) shape: {logits.shape}")
    print(f"Logits values:\n{logits}")
    
    print(f"\nFrequency Score shape: {freq_score.shape}")
    print(f"Video-level Frequency Anomaly Scores:\n{freq_score}")
    print(f"\nTemporal Score shape: {temporal_score.shape}")
    print(f"Temporal Inconsistency Scores:\n{temporal_score}")
    print("----------------------")

if __name__ == "__main__":
    main()
