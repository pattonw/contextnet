from funlib.geometry import Coordinate

from pydantic import BaseModel
import yaml

from pathlib import Path
from typing import Optional


class PydanticCoordinate(Coordinate):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Coordinate(*v)


class BackboneConfig(BaseModel):
    raw_input_channels: int
    n_output_channels: int
    num_init_features: int
    num_embeddings: int
    growth_rate: int
    block_config: list[int]
    padding: str
    embeddings: bool
    upsample_mode: str

    @property
    def context(self):
        return sum(self.block_config)


class ScaleConfig(BaseModel):
    scale_factor: PydanticCoordinate
    num_raw_scale_levels: int
    num_gt_scale_levels: int
    num_eval_scale_levels: int


class StorageConfig(BaseModel):
    dataset: str = "volumes/raw/s{level}"
    crop: str = "volumes/groundtruth/crop{crop_num}/labels/{organelle}"
    container: str = "/groups/cellmap/cellmap/data/{dataset}/{dataset}.n5"
    fallback: str = "/nrs/cellmap/pattonw/data/tmp_data/{dataset}/{dataset}.n5"


class DataSetConfig(BaseModel):
    name: str
    raw: StorageConfig = StorageConfig()
    training: list[int]
    validation: list[int]


class DataConfig(BaseModel):
    categories: list[list[int]]
    organelles: dict[str, int] = {}
    datasets: list[Path]
    min_volume_size: int
    zero_pad: int
    gt_voxel_size: PydanticCoordinate  # assumed to be half the raw voxel size

    @property
    def dataset_configs(self) -> list[DataSetConfig]:
        for dataset in self.datasets:
            yield (DataSetConfig(**yaml.safe_load(dataset.open("r").read())))


class TrainConfig(BaseModel):
    input_shape_voxels: PydanticCoordinate
    eval_input_shape_voxels: PydanticCoordinate
    num_iterations: int
    checkpoint_interval: int
    snapshot_interval: int
    validation_interval: int
    warmup: int
    batch_size: int
    num_workers: int
    checkpoint_dir: Path
    snapshot_container: Path
    validation_container: Path
    loss_file: Path
    val_file: Path
    learning_rate: float
    threshold_skew: int
    lsds: bool
    scale_config_file: Path
    data_config_file: Path
    architecture_config_file: Path
    start: Optional[Path] = None
    sample_voxel_size: bool = False
    use_organelle_vols: bool = False
    scale_as_input: bool = False

    @property
    def scale_config(self) -> ScaleConfig:
        return ScaleConfig(**yaml.safe_load(self.scale_config_file.open("r").read()))

    @property
    def data_config(self) -> DataConfig:
        return DataConfig(**yaml.safe_load(self.data_config_file.open("r").read()))

    @property
    def architecture_config(self) -> BackboneConfig:
        return BackboneConfig(
            **yaml.safe_load(self.architecture_config_file.open("r").read())
        )
