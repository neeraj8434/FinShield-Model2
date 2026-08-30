import torch
from src.model2_reference import Model

CKPT_PATH = "checkpoints/model_87_acc_20_frames_final_data.pt"

def main():
    print("Loading checkpoint...")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")

    # Some checkpoints wrap the state_dict in a dict with metadata keys
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(f"Checkpoint contains {len(state_dict)} keys")
    print("First 5 keys:", list(state_dict.keys())[:5])
    print("Last 5 keys:", list(state_dict.keys())[-5:])

    print("\nBuilding model...")
    model = Model(num_classes=2, use_frequency_feature=True)

    print("Loading state_dict with strict=False (fusion layers are new)...")
    try:
        compat = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys  (expected for new fusion layers): {compat.missing_keys}")
        print(f"Unexpected keys (should be empty):               {compat.unexpected_keys}")
        assert len(compat.unexpected_keys) == 0, (
            f"Unexpected keys found in checkpoint: {compat.unexpected_keys}"
        )
        expected_missing = {
            "fusion_fc1.weight", "fusion_fc1.bias",
            "fusion_fc2.weight", "fusion_fc2.bias",
        }
        assert set(compat.missing_keys) == expected_missing, (
            f"Missing keys mismatch!\n"
            f"  Expected: {expected_missing}\n"
            f"  Got:      {set(compat.missing_keys)}"
        )
        print("SUCCESS: all original keys loaded, only fusion layers are new.")
    except RuntimeError as e:
        print("FAILED load:")
        print(e)
        return

    model.eval()
    print("\nRunning forward pass with checkpoint weights (baseline path)...")
    mock_input = torch.rand(1, 20, 3, 112, 112)  # matches checkpoint's expected 20 frames, 112x112
    with torch.no_grad():
        fmap, logits, freq_score = model(mock_input, use_fusion_classifier=False)

    print(f"Logits: {logits}")
    probs = torch.softmax(logits, dim=-1)
    print(f"Probabilities (REAL, FAKE): {probs}")

if __name__ == "__main__":
    main()
