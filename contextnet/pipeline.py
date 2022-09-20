from .configs import DataConfig, ScaleConfig

import gunpowder as gp
import daisy
from funlib.geometry import Coordinate, Roi
from lsd.gp.add_local_shape_descriptor import AddLocalShapeDescriptor

import numpy as np

import neuroglancer
import neuroglancer.cli

from pathlib import Path

from .gp.resample import Resample
from .gp.pad import Pad
from .gp.relabel import Relabel


def build_pipeline(
    data_config: DataConfig,
    scale_config: ScaleConfig,
    gt_voxel_size: Coordinate,
    lsds: bool = False,
) -> gp.Pipeline:

    raw_scale_levels = range(scale_config.num_raw_scale_levels)
    gt_scale_levels = range(scale_config.num_gt_scale_levels)

    dataset_pipelines = []
    for dataset_config in data_config.datasets:

        training_crops = [
            crop_num
            for crop_num in dataset_config.training_crops
            if (
                dataset_config.dataset_container
                / dataset_config.gt_dataset.format(crop_num=crop_num)
            ).exists()
            or (
                dataset_config.fallback_dataset_container
                / dataset_config.gt_dataset.format(crop_num=crop_num)
            ).exists()
        ]

        # open training crops as daisy arrays
        # read from fallback location if main container does not contain
        # expected dataset
        training_datasets: list[daisy.Array] = [
            daisy.open_ds(
                f"{dataset_config.dataset_container}",
                dataset_config.gt_dataset.format(crop_num=crop_num),
            )
            if Path(
                f"{dataset_config.dataset_container}",
                dataset_config.gt_dataset.format(crop_num=crop_num),
            ).exists()
            else daisy.open_ds(
                f"{dataset_config.fallback_dataset_container}",
                dataset_config.gt_dataset.format(crop_num=crop_num),
            )
            for crop_num in training_crops
        ]

        print([training_dataset.roi.shape for training_dataset in training_datasets])

        # filter training crops by size. Since we need multiple scale
        # levels, we only want volumes large enough to train on
        training_crops, training_datasets = zip(
            *[
                (crop, dataset)
                for crop, dataset in zip(training_crops, training_datasets)
                if min(dataset.roi.shape / data_config.gt_voxel_size)
                >= data_config.min_volume_size * (2 if not lsds else 4)
                and (
                    dataset.voxel_size == gt_voxel_size
                    or dataset.voxel_size == gt_voxel_size * 2
                )
            ]
        )

        print([training_dataset.roi.shape for training_dataset in training_datasets])

        # get raw container
        raw_datasets: list[daisy.Array] = [
            daisy.open_ds(
                f"{dataset_config.dataset_container}",
                f"{dataset_config.raw_dataset}/s{scale_level}",
            )
            if Path(
                f"{dataset_config.dataset_container}",
                f"{dataset_config.raw_dataset}/s{scale_level}",
            ).exists()
            else daisy.open_ds(
                f"{dataset_config.fallback_dataset_container}",
                f"{dataset_config.raw_dataset}/s{scale_level}",
            )
            for scale_level in raw_scale_levels
        ]

        raw_scale_keys = [gp.ArrayKey(f"RAW_S{scale}") for scale in raw_scale_levels]
        labels_key = gp.ArrayKey("LABELS")
        gt_key = gp.ArrayKey("GT")
        mask_key = gp.ArrayKey("MASK")
        gt_scale_keys = [gp.ArrayKey(f"GT_S{scale}") for scale in gt_scale_levels]
        weight_scale_keys = [
            gp.ArrayKey(f"WEIGHT_S{scale}") for scale in gt_scale_levels
        ]
        mask_scale_keys = [gp.ArrayKey(f"MASK_S{scale}") for scale in gt_scale_levels]
        if lsds:
            lsd_scale_keys = [gp.ArrayKey(f"LSD_S{scale}") for scale in gt_scale_levels]
            lsd_mask_scale_keys = [
                gp.ArrayKey(f"LSD_MASK_S{scale}") for scale in gt_scale_levels
            ]

        pipeline = (
            tuple(
                (
                    gp.ZarrSource(
                        dataset.data.store.path,
                        {labels_key: dataset.data.name},
                        {
                            labels_key: gp.ArraySpec(
                                roi=dataset.roi,
                                voxel_size=dataset.voxel_size,
                                interpolatable=False,
                            )
                        },
                    )
                    + Resample(
                        labels_key,
                        gt_voxel_size,
                        labels_key,
                    )
                    + Relabel(
                        labels_key,
                        gt_key,
                        data_config.categories,
                    )
                    + Relabel(gt_key, mask_key, [[-1], [0, 1, 2, 3]])
                    + gp.Pad(gt_key, data_config.zero_pad)
                    + gp.Pad(labels_key, data_config.zero_pad)
                    + gp.Pad(mask_key, data_config.zero_pad),
                    gp.ZarrSource(
                        raw_datasets[0].data.store.path,
                        {
                            raw_scale_keys[scale_level]: raw_dataset.data.name
                            for scale_level, raw_dataset in enumerate(raw_datasets)
                        },
                        {
                            raw_scale_keys[scale_level]: gp.ArraySpec(
                                voxel_size=raw_dataset.voxel_size,
                                interpolatable=True,
                            )
                            for scale_level, raw_dataset in enumerate(raw_datasets)
                        },
                    )
                    + Pad(raw_scale_keys, None),
                )
                + gp.MergeProvider()
                + gp.RandomLocation()
                for dataset in training_datasets
            )
            + gp.RandomProvider()
        )
        dataset_pipelines.append(pipeline)
    pipeline = tuple(dataset_pipelines) + gp.RandomProvider()
    for i in gt_scale_levels:
        pipeline += Resample(
            gt_key, gt_voxel_size * (scale_config.scale_factor**i), gt_scale_keys[i]
        )
        pipeline += Resample(
            mask_key,
            gt_voxel_size * (scale_config.scale_factor**i),
            mask_scale_keys[i],
        )
        pipeline += gp.BalanceLabels(
            gt_scale_keys[i], weight_scale_keys[i], num_classes=4
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
        return (raw_arrays, gt_arrays, weight_arrays, mask_arrays, lsd_arrays, lsd_mask_arrays)

    return (raw_arrays, gt_arrays, weight_arrays, mask_arrays)
