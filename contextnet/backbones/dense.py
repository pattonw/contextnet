import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, padding):
        super().__init__()
        self.add_module(
            "conv1",
            nn.Conv3d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("relu1", nn.ReLU(inplace=False))
        self.add_module(
            "conv2",
            nn.Conv3d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=padding,
                bias=False,
            ),
        )
        self.add_module("relu2", nn.ReLU(inplace=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )

        x = x[
            (slice(None, None), slice(None, None))
            + tuple(
                slice(
                    int((x_shape - y_shape) / 2), int(y_shape + (x_shape - y_shape) / 2)
                )
                for x_shape, y_shape in zip(x.shape[2:], new_features.shape[2:])
            )
        ]
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, padding
    ):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                padding=padding,
            )
            self.add_module("denselayer{}".format(i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, padding):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv3d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
                padding=padding,
            ),
        )
        self.add_module("relu", nn.ReLU(inplace=False))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(
        self,
        n_input_channels=1,
        n_output_channels=3,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        num_embeddings=8,
        bn_size=4,
        drop_rate=0,
        padding="valid",
        upsample_mode: str = "nearest",
    ):

        super().__init__()

        self.num_embeddings = num_embeddings

        # First convolution
        first_conv = [
            (
                "conv1",
                nn.Conv3d(
                    n_input_channels,
                    num_init_features,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    padding=padding,
                ),
            ),
            ("relu1", nn.ReLU(inplace=False)),
        ]

        self.features = nn.Sequential(OrderedDict(first_conv))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                padding=padding,
            )
            self.features.add_module("denseblock{}".format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            trans = _Transition(
                num_input_features=num_features,
                num_output_features=num_init_features,
                padding=padding,
            )
            self.features.add_module("transition{}".format(i + 1), trans)
            num_features = num_init_features
        self.features.add_module(
            "embeddings", torch.nn.Conv3d(num_init_features, num_embeddings, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

        self.final_layer = nn.Conv3d(num_embeddings, n_output_channels, (1, 1, 1))
        self.upsample = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)

    def forward(self, raw, upsampled=None):
        if upsampled is None:
            upsampled_shape = list(raw.shape)
            upsampled_shape[1] = self.num_embeddings
            upsampled = torch.zeros(upsampled_shape)
        x = torch.cat([raw, upsampled], dim=1)
        features = self.upsample(self.features(x))
        out = F.relu(features, inplace=False)
        final = self.final_layer(out)
        if self.training:
            return features, final
        else:
            return features, torch.nn.Softmax(dim=1)(final)
