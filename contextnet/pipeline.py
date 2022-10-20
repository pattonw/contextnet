from .configs import DataConfig, ScaleConfig, DataSetConfig
from .gp.resample import Resample
from .gp.pad import Pad
from .gp.relabel import Relabel
from .gp.cellmap_source import CellMapSource
from .gp.scale_provider import ScaleProvider

import gunpowder as gp
import daisy
from funlib.geometry import Coordinate, Roi
from lsd.gp.add_local_shape_descriptor import AddLocalShapeDescriptor
from fibsem_tools.metadata.groundtruth import Label, LabelList

import numpy as np
import yaml
import zarr

import neuroglancer
import neuroglancer.cli

from pathlib import Path
from typing import Optional
from collections import defaultdict
from pprint import pprint


def get_datasets(dataset_config) -> list[tuple[Coordinate, dict[str, daisy.Array]]]:

    # filter out crops to which we can't find the ground truth
    def crop_exists(dataset_config, crop_num: int) -> bool:
        return (
            Path(
                dataset_config.raw.container.format(dataset=dataset_config.name),
                dataset_config.raw.crop.format(crop_num=crop_num, organelle=""),
            ).exists()
            or Path(
                dataset_config.raw.fallback.format(dataset=dataset_config.name),
                dataset_config.raw.crop.format(crop_num=crop_num, organelle=""),
            ).exists()
        )

    def get_dataset_container_group(dataset_config, crop_num):
        return (
            Path(dataset_config.raw.container.format(dataset=dataset_config.name))
            if Path(
                dataset_config.raw.container.format(dataset=dataset_config.name),
                dataset_config.raw.crop.format(crop_num=crop_num, organelle=""),
            ).exists()
            else Path(dataset_config.raw.fallback.format(dataset=dataset_config.name)),
            dataset_config.raw.crop.format(crop_num=crop_num, organelle=""),
        )

    containers = [
        get_dataset_container_group(dataset_config, crop_num)
        for crop_num in dataset_config.training
        if crop_exists(dataset_config, crop_num)
    ]

    def get_label_list(container: Path, dataset: str):
        try:
            array = zarr.open(container, "r")[dataset]
        except KeyError as e:
            print(container, dataset)
            raise e
        labels = []

        for label in array.attrs["labels"]:
            label = Label(**label)
            if label.value in labels:
                pass
            else:
                labels.append(label)

        return labels

    def get_organelle_datasets(
        container: Path, group_name: str
    ) -> tuple[Coordinate, dict[str, tuple[daisy.Array, list[Label]]]]:
        try:
            attribute_path = container / group_name / "attributes.json"
            if not attribute_path.exists():
                upstream_attribute_path = (
                    attribute_path.parent.parent.parent / "attributes.json"
                )
                if upstream_attribute_path.exists():
                    attribute_path.open("w").write(
                        upstream_attribute_path.open("r").read()
                    )
            group = zarr.open(container, "r")[group_name]
        except KeyError as e:
            print(container, group_name)
            raise e
        arrays = {}
        voxel_sizes: set[Coordinate] = set()

        def store_array(k, v):
            if isinstance(v, zarr.Array):
                array = daisy.open_ds(container, v.path)
                if len(voxel_sizes) == 0:
                    voxel_sizes.add(array.voxel_size)
                else:
                    assert array.voxel_size in voxel_sizes
                arrays[k] = (
                    array,
                    get_label_list(container, v.path),
                )

        group.visititems(store_array)
        assert len(voxel_sizes) > 0
        return list(voxel_sizes)[0], arrays

    training_datasets: list[
        tuple[Coordinate, dict[str, tuple[daisy.Array, list[Label]]]]
    ] = [get_organelle_datasets(container, group) for container, group in containers]
    return training_datasets


def build_pipeline(
    data_config: DataConfig,
    scale_config: ScaleConfig,
    lsds: bool = False,
    sample_voxel_sizes: bool = False,
    use_organelle_datasets: bool = False,
) -> gp.Pipeline:

    raw_scale_levels = range(scale_config.num_raw_scale_levels)
    gt_scale_levels = range(scale_config.num_gt_scale_levels)

    raw_scale_keys = [gp.ArrayKey(f"RAW_S{scale}") for scale in raw_scale_levels]
    gt_key = gp.ArrayKey("GT")
    mask_key = gp.ArrayKey("MASK")
    gt_scale_keys = [gp.ArrayKey(f"GT_S{scale}") for scale in gt_scale_levels]
    weight_scale_keys = [gp.ArrayKey(f"WEIGHT_S{scale}") for scale in gt_scale_levels]
    mask_scale_keys = [gp.ArrayKey(f"MASK_S{scale}") for scale in gt_scale_levels]
    if lsds:
        lsd_scale_keys = [gp.ArrayKey(f"LSD_S{scale}") for scale in gt_scale_levels]
        lsd_mask_scale_keys = [
            gp.ArrayKey(f"LSD_MASK_S{scale}") for scale in gt_scale_levels
        ]
    scale_key = gp.ArrayKey("SCALE_KEY")

    dataset_pipelines = []
    for dataset_config_path in data_config.datasets:

        dataset_config = DataSetConfig(
            **yaml.safe_load(dataset_config_path.open("r").read())
        )
        training_datasets = get_datasets(dataset_config)
        print(sample_voxel_sizes, use_organelle_datasets)
        if not sample_voxel_sizes:
            min_voxel_size = min([voxel_size for voxel_size, _ in training_datasets])
            training_datasets = [
                (voxel_size, datasets)
                for voxel_size, datasets in training_datasets
                if voxel_size == min_voxel_size
            ]
        if not use_organelle_datasets:
            training_datasets = [
                (voxel_size, {"all": datasets["all"]})
                for voxel_size, datasets in training_datasets
                if "all" in datasets
            ]

        # get raw container
        raw_default_container = Path(
            dataset_config.raw.container.format(dataset=dataset_config.name)
        )
        raw_fallback_container = Path(
            dataset_config.raw.fallback.format(dataset=dataset_config.name)
        )
        s0_dataset = dataset_config.raw.dataset.format(level=0)
        raw_container = (
            raw_default_container
            if (raw_default_container / s0_dataset).exists()
            else raw_fallback_container
        )
        raw_scale_level_datasets = {
            int(f.name[1:]): daisy.open_ds(
                f"{raw_container}", dataset_config.raw.dataset.format(level=f.name[1:])
            )
            for f in (raw_container / s0_dataset).parent.iterdir()
            if f.is_dir() and f.name.startswith("s") and f.name[1:].isnumeric()
        }
        if 0 not in raw_scale_level_datasets:
            raw_scale_level_datasets[0] = daisy.open_ds(f"{raw_container}", s0_dataset)
        raw_voxel_size_datasets = {
            array.voxel_size: array for array in raw_scale_level_datasets.values()
        }

        scale_pipelines = defaultdict(list)

        for voxel_size, datasets in training_datasets:
            if not all(
                [
                    voxel_size * 2 ** (scale_level + 1) in raw_voxel_size_datasets
                    for scale_level in raw_scale_levels
                ]
            ):
                print(
                    f"Not sufficient raw pyramid levels available for dataset: {dataset_config.name}"
                )
                continue
            pipeline = (
                (
                    CellMapSource(gt_key, mask_key, datasets, data_config)
                    + gp.Pad(gt_key, data_config.zero_pad)
                    + gp.Pad(mask_key, data_config.zero_pad),
                    gp.ZarrSource(
                        raw_scale_level_datasets[0].data.store.path,
                        {
                            raw_scale_keys[scale_level]: raw_voxel_size_datasets[
                                voxel_size * 2 ** (scale_level + 1)
                            ].data.name
                            for scale_level in raw_scale_levels
                        },
                        {
                            raw_scale_keys[scale_level]: gp.ArraySpec(
                                voxel_size=raw_voxel_size_datasets[
                                    voxel_size * 2 ** (scale_level + 1)
                                ].voxel_size,
                                interpolatable=True,
                            )
                            for scale_level in raw_scale_levels
                        },
                    )
                    + Pad(raw_scale_keys, None),
                )
                + gp.MergeProvider()
                + gp.RandomLocation()
            )
            scale_pipelines[voxel_size].append(pipeline)
        if len(scale_pipelines) > 0:
            random_scale_pipeline = tuple(
                tuple(pipelines) + gp.RandomProvider()
                for pipelines in scale_pipelines.values()
            ) + ScaleProvider(gt_key, sampled_key=scale_key)
            print("Scale pipelines: ", {k: len(v) for k, v in scale_pipelines.items()})
            dataset_pipelines.append(random_scale_pipeline)
    pipeline = tuple(dataset_pipelines) + gp.RandomProvider()
    for i in gt_scale_levels:
        base_voxel_size = Coordinate(1, 1, 1)
        pipeline += Resample(
            gt_key, base_voxel_size * (scale_config.scale_factor**i), gt_scale_keys[i]
        )
        pipeline += Resample(
            mask_key,
            base_voxel_size * (scale_config.scale_factor**i),
            mask_scale_keys[i],
        )
        pipeline += gp.BalanceLabels(
            gt_scale_keys[i],
            weight_scale_keys[i],
            mask_scale_keys[i],
            slab=(1, -1, -1, -1),
        )
        if lsds:
            pipeline += AddLocalShapeDescriptor(
                gt_scale_keys[i], lsd_scale_keys[i], lsd_mask_scale_keys[i], sigma=40
            )

    for i in raw_scale_levels:
        pipeline += gp.Normalize(raw_scale_keys[i])
        pipeline += gp.IntensityAugment(raw_scale_keys[i], 0.9, 1.1, -0.1, 0.1)
        pipeline += gp.NoiseAugment(raw_scale_keys[i], mode="gaussian", var=0.0004)
        pipeline += gp.NoiseAugment(raw_scale_keys[i], mode="s&p")

    pipeline += gp.SimpleAugment()
    return pipeline


def get_request(
    base_shape: Coordinate,
    scale_config: ScaleConfig,
    lsds: bool = False,
) -> gp.BatchRequest:
    """
    base_shape should be provided in world units. This should be the output shape
    at the highest resolution
    """

    raw_scale_levels = range(scale_config.num_raw_scale_levels)
    gt_scale_levels = range(scale_config.num_gt_scale_levels)

    raw_shapes = [
        base_shape * (scale_config.scale_factor**scale_level)
        for scale_level in raw_scale_levels
    ]
    gt_shapes = [
        base_shape * (scale_config.scale_factor**scale_level)
        for scale_level in gt_scale_levels
    ]

    raw_scale_keys = [gp.ArrayKey(f"RAW_S{scale}") for scale in range(len(raw_shapes))]
    gt_scale_keys = [gp.ArrayKey(f"GT_S{scale}") for scale in range(len(gt_shapes))]
    weight_scale_keys = [
        gp.ArrayKey(f"WEIGHT_S{scale}") for scale in range(len(gt_shapes))
    ]
    mask_scale_keys = [gp.ArrayKey(f"MASK_S{scale}") for scale in gt_scale_levels]
    scale_key = gp.ArrayKey("SCALE_KEY")

    if lsds:
        lsd_scale_keys = [gp.ArrayKey(f"LSD_S{scale}") for scale in gt_scale_levels]
        lsd_mask_scale_keys = [
            gp.ArrayKey(f"LSD_MASK_S{scale}") for scale in gt_scale_levels
        ]

    request = gp.BatchRequest()
    for raw_key, input_shape in zip(raw_scale_keys, raw_shapes):
        request.add(
            raw_key,
            input_shape,
        )
    for gt_key, weight_key, mask_key, output_shape in zip(
        gt_scale_keys, weight_scale_keys, mask_scale_keys, gt_shapes
    ):
        request.add(
            gt_key,
            output_shape,
        )
        request.add(
            weight_key,
            output_shape,
        )
        request.add(
            mask_key,
            output_shape,
        )
    if lsds:
        for lsd_key, lsd_mask_key, output_shape in zip(
            lsd_scale_keys, lsd_mask_scale_keys, gt_shapes
        ):
            request.add(lsd_key, output_shape)
            request.add(lsd_mask_key, output_shape)
    request[scale_key] = gp.ArraySpec(nonspatial=True)
    return request


def split_batch(
    batch: gp.BatchRequest, scale_config: ScaleConfig, lsds: bool = False
) -> tuple[list[gp.Array], list[gp.Array], list[gp.Array]]:
    """
    base_shape should be provided in world units. This should be the output shape
    at the highest resolution
    """

    raw_scale_levels = range(scale_config.num_raw_scale_levels)
    gt_scale_levels = range(scale_config.num_gt_scale_levels)

    raw_scale_keys = [gp.ArrayKey(f"RAW_S{scale}") for scale in raw_scale_levels]
    gt_scale_keys = [gp.ArrayKey(f"GT_S{scale}") for scale in gt_scale_levels]
    weight_scale_keys = [gp.ArrayKey(f"WEIGHT_S{scale}") for scale in gt_scale_levels]
    mask_scale_keys = [gp.ArrayKey(f"MASK_S{scale}") for scale in gt_scale_levels]
    scale_key = gp.ArrayKey("SCALE_KEY")

    if lsds:
        lsd_scale_keys = [gp.ArrayKey(f"LSD_S{scale}") for scale in gt_scale_levels]
        lsd_mask_scale_keys = [
            gp.ArrayKey(f"LSD_MASK_S{scale}") for scale in gt_scale_levels
        ]

    raw_arrays = [batch[key] for key in raw_scale_keys]
    gt_arrays = [batch[key] for key in gt_scale_keys]
    weight_arrays = [batch[key] for key in weight_scale_keys]
    mask_arrays = [batch[key] for key in mask_scale_keys]
    if lsds:
        lsd_arrays = [batch[key] for key in lsd_scale_keys]
        lsd_mask_arrays = [batch[key] for key in lsd_mask_scale_keys]
        return (
            raw_arrays,
            gt_arrays,
            weight_arrays,
            mask_arrays,
            lsd_arrays,
            lsd_mask_arrays,
            batch[scale_key],
        )

    return (raw_arrays, gt_arrays, weight_arrays, mask_arrays, batch[scale_key])
