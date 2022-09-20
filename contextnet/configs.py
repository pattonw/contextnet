from funlib.geometry import Coordinate

from pydantic import BaseModel

from pathlib import Path


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


class ScaleConfig(BaseModel):
    scale_factor: PydanticCoordinate
    num_raw_scale_levels: int
    num_gt_scale_levels: int


class DataSetConfig(BaseModel):
    dataset_container: Path
    fallback_dataset_container: Path
    raw_dataset: str
    gt_group: str
    training_crops: list[int]
    validation_crops: list[int]


class DataConfig(BaseModel):
    datasets: list[DataSetConfig]
    min_volume_size: int
    zero_pad: int
    gt_voxel_size: PydanticCoordinate  # assumed to be half the raw voxel size
    categories: list[list[int]]
    organelles: dict[str, int] = {}
