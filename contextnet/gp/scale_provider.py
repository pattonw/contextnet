import copy
import numpy as np

from gunpowder import BatchProvider

import random


class ScaleProvider(BatchProvider):
    """Selects one of the upstream providers::

        (a + b + c) + ScaleProvider()

    based on the voxel_size of a specific key in the upstream providers.
    """

    def __init__(self, scale_key):
        self.scale_key = scale_key

    def setup(self):
        self.enable_placeholders()
        assert (
            len(self.get_upstream_providers()) > 0
        ), "at least one batch provider must be added to the ScaleProvider"

        common_spec = None

        # advertise outputs only if all upstream providers have them
        self.scale_providers = {}
        for provider in self.get_upstream_providers():
            assert (
                self.scale_key in provider.spec
            ), f"All providers must provide {self.scale_key}"
            provider_scale = provider.spec[self.scale_key].voxel_size
            assert (
                provider_scale not in self.scale_providers
            ), f"All providers must provide {self.scale_key} with unique voxel size"
            self.scale_providers[provider_scale] = provider

            if common_spec is None:
                common_spec = copy.deepcopy(provider.spec)
            else:
                for key, spec in list(common_spec.items()):
                    if key not in provider.spec:
                        del common_spec[key]

        for key, spec in common_spec.items():
            spec.voxel_size = None
            self.provides(key, spec)

        self.resolutions = [
            provider.spec[self.scale_key].voxel_size
            for provider in self.get_upstream_providers()
        ]

    def provide(self, request):
        resolution = random.choice(self.resolutions)
        provider = self.scale_providers[resolution]
        for key, spec in request.items():
            spec.roi.offset *= resolution
            spec.roi.shape *= resolution
        batch = provider.request_batch(request)
        for key, array in batch.arrays.items():
            array.spec.voxel_size /= resolution
            array.spec.roi.offset /= resolution
            array.spec.roi.shape /= resolution
        return batch
