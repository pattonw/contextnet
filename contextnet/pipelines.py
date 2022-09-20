from .configs import DataConfig, ScaleConfig

import gunpowder as gp
import daisy
from funlib.geometry import Coordinate, Roi
from lsd.gp.add_local_shape_descriptor import AddLocalShapeDescriptor

import numpy as np
import zarr

from pathlib import Path
from collections import defaultdict
import copy
import io

from .gp.resample import Resample
from .gp.pad import Pad
from .gp.binarize import Binarize
from .gp.scale_provider import ScaleProvider
from .gp.elastic_augment import ElasticAugment


class ArraySource(gp.BatchProvider):
    def __init__(self, key: gp.ArrayKey, array: gp.Array):
        self.key = key
        self.array = array

    def setup(self):
        self.provides(self.key, self.array.spec.copy())

    def provide(self, request):
        outputs = gp.Batch()
        outputs[self.key] = copy.deepcopy(self.array.crop(request[self.key].roi))
        return outputs


class GTMaskSource(gp.BatchProvider):
    def __init__(self, gt_key, mask_key, datasets, data_config):
        self.gt_key = gt_key
        self.mask_key = mask_key
        self.datasets = datasets
        self.data_config = data_config

        self.voxel_size = list(datasets.values())[0].voxel_size
        self.roi = list(datasets.values())[0].roi

    def setup(self):
        self.provides(
            self.gt_key, gp.ArraySpec(self.roi, self.voxel_size, False, dtype=np.uint8)
        )
        self.provides(
            self.mask_key,
            gp.ArraySpec(self.roi, self.voxel_size, False, dtype=np.uint8),
        )

    def provide(self, request):

        crop_data = None
        crop_mask = None
        roi = request[self.gt_key].roi
        for organelle, dataset in self.datasets.items():
            if crop_data is None:
                crop_data = np.zeros(
                    (len(self.data_config.categories) + 1,)
                    + dataset.to_ndarray(roi).shape,
                    dtype=np.uint8,
                )
                crop_mask = np.zeros(
                    (len(self.data_config.categories) + 1,)
                    + dataset.to_ndarray(roi).shape,
                    dtype=np.uint8,
                )

            if organelle == "all":
                data = dataset.to_ndarray(roi)
                crop_data = np.stack(
                    [np.isin(data, group) for group in self.data_config.categories]
                )
                crop_data = np.stack([crop_data.sum(axis=0) == 0, *crop_data])
                crop_mask = np.ones_like(crop_data)

            else:
                if organelle not in self.data_config.organelles:
                    continue
                # mask in all annotated data
                mask = dataset.to_ndarray(roi) > 0
                # this region can be masked in for the background channel
                crop_mask[0] = np.stack([crop_mask[0], mask]).max(axis=0)
                # get channel associated with this organelle
                channel = self.data_config.organelles[organelle]
                # get ids associated with this channel
                ids = set(self.data_config.categories[channel])
                # mask in/out data in other channels based on this organelle
                for group_channel, group in enumerate(self.data_config.categories):
                    # if all group ids are contained in ids we can mask in
                    # negatives or everything
                    if all([g_id in ids for g_id in group]):
                        # if this channels ids perfectly match organelle ids
                        # mask in everything, else mask in negatives
                        if len(ids) == len(group):
                            crop_data[group_channel + 1] = mask
                            crop_mask[group_channel + 1] = 1
                        else:
                            crop_mask[group_channel + 1] = np.max(
                                np.stack((1 - mask, crop_mask[group_channel + 1])),
                                axis=0,
                            )
                    elif all([g_id not in ids for g_id in group]):
                        # if there is no overlap we can mask in positives
                        group_mask = crop_mask[group_channel + 1]
                        stack_mask = np.stack((mask, group_mask))
                        intersect_mask = np.max(stack_mask, axis=0)
                        crop_mask[group_channel + 1] = intersect_mask
                    else:
                        # if there is some overlap between ids and group, we
                        # don't know anything.
                        pass
                # mask in everything for this channel
                crop_data[channel + 1] = mask
                crop_mask[channel + 1] = 1

        # background channel. All foreground regions are already masked in, keep them.
        # any voxel that is background in all other channels can be kept.
        crop_mask[0] = np.stack([crop_mask[0], crop_mask[1:].min(axis=0)]).max(axis=0)
        # set groundtruth background. anywhere nothing else is foreground.
        crop_data[0] = np.stack([crop_data[0], 1 - crop_data[1:].max(axis=0)]).max(axis=0)
        output = gp.Batch()
        output[self.gt_key] = gp.Array(
            crop_data.astype(np.uint8), gp.ArraySpec(roi, self.voxel_size, False)
        )
        output[self.mask_key] = gp.Array(
            crop_mask.astype(np.uint8), gp.ArraySpec(roi, self.voxel_size, False)
        )
        return output


def source_from_dataset(
    datasets,
    labels_key,
    gt_key,
    weight_key,
    data_config,
    resolution,
    gt_scale_keys,
    mask_scale_keys,
):

    source = (
        GTMaskSource(
            gt_key,
            weight_key,
            datasets,
            data_config,
        )
        + Resample(
            gt_key,
            resolution,
            gt_scale_keys[0],
        )
        + Resample(
            gt_key,
            resolution * 2,
            gt_scale_keys[1],
        )
        + gp.Pad(gt_scale_keys[0], resolution * data_config.zero_pad)
        + gp.Pad(gt_scale_keys[1], resolution * data_config.zero_pad * 2)
        + Resample(
            weight_key,
            resolution,
            mask_scale_keys[0],
        )
        + Resample(
            weight_key,
            resolution * 2,
            mask_scale_keys[1],
        )
        + gp.Pad(mask_scale_keys[0], resolution * data_config.zero_pad)
        + gp.Pad(mask_scale_keys[1], resolution * data_config.zero_pad * 2)
    )
    return source


def get_datasets(dataset_config) -> list[dict[str, daisy.Array]]:

    # filter out crops to which we can't find the ground truth
    def crop_exists(dataset_config, crop_num: int) -> bool:
        return (
            dataset_config.dataset_container
            / dataset_config.gt_group.format(crop_num=crop_num)
        ).exists() or (
            dataset_config.fallback_dataset_container
            / dataset_config.gt_group.format(crop_num=crop_num)
        ).exists()

    def get_dataset_container_group(dataset_config, crop_num):
        return (
            dataset_config.dataset_container
            if (
                dataset_config.dataset_container
                / dataset_config.gt_group.format(crop_num=crop_num)
            ).exists()
            else dataset_config.fallback_dataset_container,
            dataset_config.gt_group.format(crop_num=crop_num),
        )

    containers = [
        get_dataset_container_group(dataset_config, crop_num)
        for crop_num in dataset_config.training_crops
        if crop_exists(dataset_config, crop_num)
    ]

    def get_organelle_datasets(container: Path, group_name: str):
        try:
            group = zarr.open(container, "r")[group_name]
        except KeyError as e:
            print(container, group)
            raise e
        arrays = {}

        def store_array(k, v):
            if isinstance(v, zarr.Array):
                arrays[k] = daisy.open_ds(v.store.path, v.path)

        group.visititems(store_array)
        return arrays

    training_datasets: list[dict[str, daisy.Array]] = [
        get_organelle_datasets(container, group) for container, group in containers
    ]

    return training_datasets


def build_pipeline(
    data_config: DataConfig,
    scale_config: ScaleConfig,
    gt_voxel_size: Coordinate,
    lsds: bool = False,
) -> gp.Pipeline:

    raw_scale_levels = range(scale_config.num_raw_scale_levels)
    gt_scale_levels = range(scale_config.num_gt_scale_levels)

    resolution_pipelines = defaultdict(lambda: list())
    for dataset_config in data_config.datasets:

        training_datasets = get_datasets(dataset_config)
        # open training crops as daisy arrays
        # read from fallback location if main container does not contain
        # expected dataset

        # sort training crops into resolutions at which they can be trained
        resolutions: dict[Coordinate, list[dict[str, daisy.Array]]] = defaultdict(
            lambda: list()
        )
        for datasets in training_datasets:
            organelles = datasets.keys()
            if "all" in organelles:
                assert len(organelles) == 1
            for gt_scale_level in gt_scale_levels:
                resolution = None
                group = {}
                for organelle, dataset in datasets.items():
                    if resolution is None:
                        resolution = dataset.voxel_size * 2 ** (gt_scale_level)
                    else:
                        assert resolution == dataset.voxel_size * 2 ** (gt_scale_level)
                    dataset_roi = dataset.roi.snap_to_grid(resolution, mode="shrink")
                    # we can train on this dataset at this resolution if shape at this
                    # resolution is twice some minimum size and the dataset resolution
                    # is higher (i.e. the dataset doesn't need to be upsampled)
                    if min(dataset_roi.shape / resolution) > (
                        data_config.min_volume_size * 2
                    ) and (min(dataset.voxel_size) <= min(resolution)):
                        group[organelle] = dataset
                if len(group) > 0:
                    resolutions[resolution].append(group)

        print(
            "Num crops/res:",
            {resolution: len(crops) for resolution, crops in resolutions.items()},
        )

        raw_group_path = Path(
            f"{dataset_config.dataset_container}/{dataset_config.raw_dataset}"
        )
        raw_group_fallback_path = Path(
            f"{dataset_config.fallback_dataset_container}/{dataset_config.raw_dataset}"
        )
        raw_scale_levels = [
            int(ds.name[1:])
            for ds in (
                raw_group_path if raw_group_path.exists else raw_group_fallback_path
            ).iterdir()
            if ds.is_dir()
        ]

        # get raw container
        raw_dataset_list = [
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
        raw_datasets: dict[Coordinate, daisy.Array] = {
            dataset.voxel_size: dataset for dataset in raw_dataset_list
        }

        raw_scale_keys = [
            gp.ArrayKey(f"RAW_HIGH"),
            gp.ArrayKey(f"RAW_LOW"),
            gp.ArrayKey(f"RAW_CONTEXT"),
        ]
        labels_key = gp.ArrayKey("LABELS")
        gt_key = gp.ArrayKey("GT")
        mask_key = gp.ArrayKey("MASK")
        gt_scale_keys = [gp.ArrayKey(f"GT_HIGH"), gp.ArrayKey(f"GT_LOW")]
        weight_scale_keys = [gp.ArrayKey(f"WEIGHT_HIGH"), gp.ArrayKey(f"WEIGHT_LOW")]
        mask_scale_keys = [gp.ArrayKey(f"MASK_HIGH"), gp.ArrayKey(f"MASK_LOW")]

        for resolution, crops in resolutions.items():

            pipeline = (
                tuple(
                    (
                        source_from_dataset(
                            datasets,
                            labels_key,
                            gt_key,
                            mask_key,
                            data_config,
                            resolution,
                            gt_scale_keys,
                            mask_scale_keys,
                            # weight_scale_keys,
                        ),
                        gp.ZarrSource(
                            raw_datasets[resolution * 2].data.store.path,
                            {
                                raw_scale_keys[0]: raw_datasets[
                                    resolution * 2
                                ].data.name,
                                raw_scale_keys[1]: raw_datasets[
                                    resolution * 4
                                ].data.name,
                                raw_scale_keys[2]: raw_datasets[
                                    resolution * 8
                                ].data.name,
                            },
                            {
                                raw_scale_keys[0]: gp.ArraySpec(
                                    voxel_size=resolution * 2,
                                    interpolatable=True,
                                ),
                                raw_scale_keys[1]: gp.ArraySpec(
                                    voxel_size=resolution * 4,
                                    interpolatable=True,
                                ),
                                raw_scale_keys[2]: gp.ArraySpec(
                                    voxel_size=resolution * 8,
                                    interpolatable=True,
                                ),
                            },
                        )
                        + Pad(raw_scale_keys, None),
                    )
                    + gp.MergeProvider()
                    + gp.RandomLocation()
                    + ElasticAugment(
                        (10, 10, 10),
                        (3, 3, 3),
                        rotation_interval=(0, 2),
                        subsample=4,
                        uniform_3d_rotation=True,
                    )
                    for datasets in crops
                )
                + gp.RandomProvider()
            )

            for raw_key in raw_scale_keys:
                pipeline += gp.Normalize(raw_key)
                pipeline += gp.IntensityAugment(raw_key, 0.9, 1.1, -0.1, 0.1)
                pipeline += gp.NoiseAugment(raw_key, mode="gaussian", var=0.0004)
                pipeline += gp.NoiseAugment(raw_key, mode="s&p")
            pipeline += gp.BalanceLabels(
                gt_scale_keys[0],
                weight_scale_keys[0],
                mask_scale_keys[0],
                slab=(1, -1, -1, -1),
            )
            pipeline += gp.BalanceLabels(
                gt_scale_keys[1],
                weight_scale_keys[1],
                mask_scale_keys[1],
                slab=(1, -1, -1, -1),
            )

            pipeline += gp.SimpleAugment()

            resolution_pipelines[resolution].append(pipeline)

    scale_pipelines = tuple(
        tuple(dataset_pipelines) + gp.RandomProvider()
        for resolution, dataset_pipelines in resolution_pipelines.items()
    )

    scale_pipeline = scale_pipelines + ScaleProvider(scale_key=gt_scale_keys[0])

    return scale_pipeline


def get_request(
    base_shape: Coordinate,
    scale_config: ScaleConfig,
    lsds: bool = False,
) -> gp.BatchRequest:
    """
    base_shape should be provided in world units. This should be the output shape
    at the highest resolution
    """

    raw_shapes = [
        base_shape,
        base_shape * scale_config.scale_factor,
        base_shape * scale_config.scale_factor**2,
    ]
    gt_shapes = [
        base_shape,
        base_shape * scale_config.scale_factor,
    ]

    raw_scale_keys = [
        gp.ArrayKey(f"RAW_HIGH"),
        gp.ArrayKey(f"RAW_LOW"),
        gp.ArrayKey(f"RAW_CONTEXT"),
    ]
    labels_key = gp.ArrayKey("LABELS")
    gt_key = gp.ArrayKey("GT")
    gt_scale_keys = [gp.ArrayKey(f"GT_HIGH"), gp.ArrayKey(f"GT_LOW")]
    weight_scale_keys = [gp.ArrayKey(f"WEIGHT_HIGH"), gp.ArrayKey(f"WEIGHT_LOW")]
    if lsds:
        lsd_scale_keys = [gp.ArrayKey(f"LSD_HIGH"), gp.ArrayKey(f"LSD_LOW")]
        lsd_mask_scale_keys = [
            gp.ArrayKey(f"LSD_MASK_HIGH"),
            gp.ArrayKey(f"LSD_MASK_LOW"),
        ]

    request = gp.BatchRequest()
    for raw_key, input_shape in zip(raw_scale_keys, raw_shapes):
        request.add(
            raw_key,
            input_shape,
        )
    for gt_key, weight_key, output_shape in zip(
        gt_scale_keys, weight_scale_keys, gt_shapes
    ):
        request.add(
            gt_key,
            output_shape,
        )
        request.add(
            weight_key,
            output_shape,
        )
    if lsds:
        for lsd_key, lsd_mask_key, output_shape in zip(
            lsd_scale_keys, lsd_mask_scale_keys, gt_shapes
        ):
            request.add(lsd_key, output_shape)
            request.add(lsd_mask_key, output_shape)
    request[gt_scale_keys[0]]
    return request


def split_batch(
    batch: gp.BatchRequest, scale_config: ScaleConfig, lsds: bool = False
) -> tuple[list[gp.Array], list[gp.Array], list[gp.Array]]:
    """
    base_shape should be provided in world units. This should be the output shape
    at the highest resolution
    """

    raw_scale_keys = [
        gp.ArrayKey(f"RAW_HIGH"),
        gp.ArrayKey(f"RAW_LOW"),
        gp.ArrayKey(f"RAW_CONTEXT"),
    ]
    gt_scale_keys = [gp.ArrayKey(f"GT_HIGH"), gp.ArrayKey(f"GT_LOW")]
    weight_scale_keys = [gp.ArrayKey(f"WEIGHT_HIGH"), gp.ArrayKey(f"WEIGHT_LOW")]
    if lsds:
        lsd_scale_keys = [gp.ArrayKey(f"LSD_HIGH"), gp.ArrayKey(f"LSD_LOW")]
        lsd_mask_scale_keys = [
            gp.ArrayKey(f"LSD_MASK_HIGH"),
            gp.ArrayKey(f"LSD_MASK_LOW"),
        ]

    raw_arrays = [batch[key] for key in raw_scale_keys]
    gt_arrays = [batch[key] for key in gt_scale_keys]
    weight_arrays = [batch[key] for key in weight_scale_keys]
    if lsds:
        lsd_arrays = [batch[key] for key in lsd_scale_keys]
        lsd_mask_arrays = [batch[key] for key in lsd_mask_scale_keys]
        return (raw_arrays, gt_arrays, weight_arrays, lsd_arrays, lsd_mask_arrays)

    return (raw_arrays, gt_arrays, weight_arrays)
