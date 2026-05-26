import pytest

torch = pytest.importorskip("torch")

from model import SliceAttentionClassifier


def test_slice_attention_classifier_outputs_expected_shapes() -> None:
    model = SliceAttentionClassifier(num_classes=5)
    x = torch.rand(2, 3, 1, 32, 32)

    logits, attention_weights, slice_features = model(x)

    assert logits.shape == (2, 5)
    assert attention_weights.shape == (2, 3)
    assert slice_features.shape == (2, 3, 128)
    torch.testing.assert_close(attention_weights.sum(dim=1), torch.ones(2))


def test_slice_attention_classifier_rejects_invalid_input_shape() -> None:
    model = SliceAttentionClassifier(num_classes=5)

    with pytest.raises(ValueError, match="Expected input"):
        model(torch.rand(3, 1, 32, 32))
