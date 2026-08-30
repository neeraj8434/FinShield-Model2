"""Standalone test: exercise both fusion and baseline forward paths.

Run:  python test_fusion_paths.py
"""
import torch
from src.model2_reference import Model


def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_length = 5
    c, h, w = 3, 224, 224
    mock_input = torch.rand(batch_size, seq_length, c, h, w)

    model = Model(num_classes=2, use_frequency_feature=True)
    model.eval()

    # ── Fusion path (default) ────────────────────────────────────────────
    print("=" * 60)
    print("PATH 1: use_fusion_classifier=True  (feature-level fusion)")
    print("=" * 60)
    with torch.no_grad():
        fmap, logits, freq_score, temporal_score = model(
            mock_input, use_fusion_classifier=True
        )
    print(f"  fmap shape           : {fmap.shape}")
    print(f"  logits shape         : {logits.shape}")
    print(f"  logits values        : {logits}")
    print(f"  freq_score shape     : {freq_score.shape}")
    print(f"  freq_score values    : {freq_score}")
    print(f"  temporal_score shape : {temporal_score.shape}")
    print(f"  temporal_score values: {temporal_score}")

    assert logits.shape == (batch_size, 2), f"Expected logits shape ({batch_size}, 2), got {logits.shape}"
    assert freq_score.shape == (batch_size,), f"Expected freq_score shape ({batch_size},), got {freq_score.shape}"
    assert temporal_score.shape == (batch_size,), f"Expected temporal_score shape ({batch_size},), got {temporal_score.shape}"
    print("  ✓ All shapes correct\n")

    # ── Baseline path ────────────────────────────────────────────────────
    print("=" * 60)
    print("PATH 2: use_fusion_classifier=False  (baseline, no fusion)")
    print("=" * 60)
    with torch.no_grad():
        fmap_b, logits_b, freq_score_b = model(
            mock_input, use_fusion_classifier=False
        )
    print(f"  fmap shape           : {fmap_b.shape}")
    print(f"  logits shape         : {logits_b.shape}")
    print(f"  logits values        : {logits_b}")
    print(f"  freq_score shape     : {freq_score_b.shape}")
    print(f"  freq_score values    : {freq_score_b}")

    assert logits_b.shape == (batch_size, 2), f"Expected logits shape ({batch_size}, 2), got {logits_b.shape}"
    assert freq_score_b.shape == (batch_size,), f"Expected freq_score shape ({batch_size},), got {freq_score_b.shape}"
    print("  ✓ All shapes correct\n")

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Fusion logits  : {logits}")
    print(f"  Baseline logits: {logits_b}")
    print(f"  (Values differ because fusion path goes through fusion_fc1/fc2,")
    print(f"   baseline path goes through linear1.)")
    print("\n  ✅ Both code paths executed without errors.")


if __name__ == "__main__":
    main()
