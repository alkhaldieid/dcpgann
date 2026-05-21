from pathlib import Path

from PIL import Image

from dcpgann.data import DataConfig, IDCDataModule, labels_from_dataset


def _make_tiny_idc(root: Path, per_class: int = 12) -> None:
    for label in ["0", "1"]:
        class_dir = root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        value = 30 if label == "0" else 220
        for idx in range(per_class):
            image = Image.new("RGB", (50, 50), color=(value, value, value))
            image.save(class_dir / f"patient{idx:03d}_idx5_x0_y0_class{label}.png")


def test_datamodule_creates_reproducible_stratified_splits(tmp_path: Path) -> None:
    data_dir = tmp_path / "idc"
    split_file = tmp_path / "splits.json"
    _make_tiny_idc(data_dir)

    config = DataConfig(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0,
        val_split=0.25,
        test_split=0.25,
        image_size=32,
        seed=13,
        split_file=split_file,
    )
    first = IDCDataModule(config).setup()
    second = IDCDataModule(config).setup()

    assert split_file.exists()
    assert first.splits == second.splits
    assert len(first.train.dataset) == 12
    assert len(first.val.dataset) == 6
    assert len(first.test.dataset) == 6
    assert set(labels_from_dataset(first.val.dataset).tolist()) == {0, 1}
