from pathlib import Path

from PIL import Image

from make_image2d_class import create_image2d_class


def test_create_image2d_class_keeps_one_informative_gate(tmp_path: Path) -> None:
    source_dir = tmp_path / "dataset" / "chair"
    source_dir.mkdir(parents=True)
    for sample_idx in range(2):
        for gate_idx in range(3):
            value = 50 + sample_idx * 20 + gate_idx
            Image.new("L", (4, 4), color=value).save(source_dir / f"chair_{sample_idx:04d}_gate_{gate_idx}.png")

    target_dir = create_image2d_class(
        dataset_root=tmp_path / "dataset",
        source_classes=["chair"],
        target_class="image2d",
        num_samples=2,
        seed=7,
        informative_gate=1,
    )

    assert (target_dir / "manifest.csv").exists()
    samples = sorted(target_dir.glob("*_gate_0.png"))
    assert len(samples) == 2

    for sample_path in samples:
        sample_id = sample_path.stem.rsplit("_gate_", 1)[0]
        gate_0 = Image.open(target_dir / f"{sample_id}_gate_0.png")
        gate_1 = Image.open(target_dir / f"{sample_id}_gate_1.png")
        gate_2 = Image.open(target_dir / f"{sample_id}_gate_2.png")

        assert gate_0.getextrema() == (0, 0)
        assert gate_1.getextrema()[1] > 0
        assert gate_2.getextrema() == (0, 0)
