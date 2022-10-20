from contextnet.configs import DataConfig

import gunpowder as gp
import daisy
from fibsem_tools.metadata.groundtruth import Label

import numpy as np

from pprint import pprint


class CellMapSource(gp.BatchProvider):
    """
    Provides gt and mask for a cellmap crop.
    Cellmap crops come in a few forms. labels/all contains semantic labels for
    38 classes of organelles
    labels/{organelle} contains instance labels for a specific label or set of
    labels
    """

    def __init__(
        self,
        gt_key,
        mask_key,
        datasets: dict[str, tuple[daisy.Array, list[Label]]],
        data_config: DataConfig,
    ):
        self.gt_key = gt_key
        self.mask_key = mask_key
        self.datasets = datasets
        self.data_config = data_config

        self.voxel_size = list(datasets.values())[0][0].voxel_size
        self.roi = list(datasets.values())[0][0].roi

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
        for organelle, (dataset, labels) in self.datasets.items():
            if crop_data is None:
                crop_data = np.zeros(
                    (len(self.data_config.categories),) + dataset.to_ndarray(roi).shape,
                    dtype=np.uint8,
                )
                crop_mask = np.zeros(
                    (len(self.data_config.categories),) + dataset.to_ndarray(roi).shape,
                    dtype=np.uint8,
                )

            if organelle == "all":
                data = dataset.to_ndarray(roi)
                crop_data = np.stack(
                    [np.isin(data, group) for group in self.data_config.categories]
                )
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
                            crop_data[group_channel] = mask
                            crop_mask[group_channel] = 1
                        else:
                            crop_mask[group_channel] = np.max(
                                np.stack((1 - mask, crop_mask[group_channel])),
                                axis=0,
                            )
                    elif all([g_id not in ids for g_id in group]):
                        # if there is no overlap we can mask in positives
                        group_mask = crop_mask[group_channel]
                        stack_mask = np.stack((mask, group_mask))
                        intersect_mask = np.max(stack_mask, axis=0)
                        crop_mask[group_channel] = intersect_mask
                    else:
                        # if there is some overlap between ids and group, we
                        # don't know anything.
                        pass
                # mask in everything for this channel
                crop_data[channel] = mask
                crop_mask[channel] = 1

        output = gp.Batch()
        output[self.gt_key] = gp.Array(
            crop_data.astype(np.uint8), gp.ArraySpec(roi, self.voxel_size, False)
        )
        output[self.mask_key] = gp.Array(
            crop_mask.astype(np.uint8), gp.ArraySpec(roi, self.voxel_size, False)
        )
        return output
