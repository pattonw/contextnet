import gunpowder as gp

import numpy as np


class Binarize(gp.BatchFilter):
    def __init__(self, labels, relabels, mapping, dtype=np.uint8):
        self.labels = labels
        self.relabels = relabels
        self.mapping = mapping
        self.dtype = dtype

    def setup(self):
        relabel_spec = self.spec[self.labels].copy()
        relabel_spec.dtype = self.dtype
        self.provides(self.relabels, relabel_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.labels] = request[self.relabels].copy()
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        labels = batch[self.labels]
        relabel_data = np.stack([np.isin(labels.data, group) for group in self.mapping])
        relabel_data = np.concatenate(
            [relabel_data.sum(axis=0, keepdims=True) == 0, relabel_data], axis=0
        ).astype(self.dtype)

        relabel_spec = labels.spec.copy()
        relabel_spec.dtype = self.dtype
        outputs[self.relabels] = gp.Array(relabel_data, relabel_spec)

        return outputs
