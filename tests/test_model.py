import pytest

torch = pytest.importorskip("torch")

from model import SliceAttentionClassifier


@pytest.mark.parametrize("fusion_mode", ["attention", "mean", "concat", "attention_residual"])
def test_slice_attention_classifier_outputs_expected_shapes(fusion_mode: str) -> None:
    model = SliceAttentionClassifier(num_classes=5, fusion_mode=fusion_mode, num_slices=3)
    x = torch.rand(2, 3, 1, 32, 32)

    logits, attention_weights, slice_features = model(x)

    assert logits.shape == (2, 5)
    assert attention_weights.shape == (2, 3)
    assert slice_features.shape == (2, 3, 128)
    torch.testing.assert_close(attention_weights.sum(dim=1), torch.ones(2))


@pytest.mark.parametrize("fusion_mode", ["mean", "concat"])
def test_non_attention_fusion_returns_uniform_weights(fusion_mode: str) -> None:
    model = SliceAttentionClassifier(num_classes=5, fusion_mode=fusion_mode, num_slices=3)

    _, attention_weights, _ = model(torch.rand(2, 3, 1, 32, 32))

    torch.testing.assert_close(attention_weights, torch.full((2, 3), 1.0 / 3.0))


def test_concat_fusion_requires_num_slices() -> None:
    with pytest.raises(ValueError, match="num_slices"):
        SliceAttentionClassifier(num_classes=5, fusion_mode="concat")


def test_attention_residual_fusion_requires_num_slices() -> None:
    with pytest.raises(ValueError, match="num_slices"):
        SliceAttentionClassifier(num_classes=5, fusion_mode="attention_residual")


def test_attention_residual_fusion_rejects_unexpected_slice_count() -> None:
    model = SliceAttentionClassifier(num_classes=5, fusion_mode="attention_residual", num_slices=3)

    with pytest.raises(ValueError, match="Expected 3 slices"):
        model(torch.rand(2, 2, 1, 32, 32))


def test_concat_fusion_rejects_unexpected_slice_count() -> None:
    model = SliceAttentionClassifier(num_classes=5, fusion_mode="concat", num_slices=3)

    with pytest.raises(ValueError, match="Expected 3 slices"):
        model(torch.rand(2, 2, 1, 32, 32))


def test_slice_attention_classifier_rejects_invalid_input_shape() -> None:
    model = SliceAttentionClassifier(num_classes=5)

    with pytest.raises(ValueError, match="Expected input"):
        model(torch.rand(3, 1, 32, 32))
