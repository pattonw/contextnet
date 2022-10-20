import click


@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("--weights/--no-weights", type=bool, default=False)
@click.option("--loss/--no-loss", type=bool, default=False)
@click.option("--argmax/--no-argmax", type=bool, default=False)
def view_snapshots(train_config, weights, loss, argmax):
    from contextnet.configs import ScaleConfig, TrainConfig

    import daisy

    import neuroglancer
    import yaml
    import numpy as np

    from itertools import chain

    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))

    neuroglancer.set_server_bind_address("0.0.0.0")

    viewer = neuroglancer.Viewer()

    raw_datasets = [
        "raw_high",
        "raw_low",
        "raw_context",
    ]
    target_datasets = [
        "target_high",
        "target_low",
    ]
    weight_datasets = [
        "weight_high",
        "weight_low",
    ]
    pred_datasets = [
        "pred_high",
        "pred_low",
        "pred_context",
    ]

    with viewer.txn() as s:
        while len(s.layers) > 0:
            del s.layers[0]
        for dataset in chain(
            raw_datasets, target_datasets, weight_datasets, pred_datasets
        ):
            daisy_array = daisy.open_ds(
                train_config.snapshot_container,
                f"{dataset}",
            )
            ndims = len(daisy_array.data.shape)
            if ndims == 4 or (ndims == 5 and argmax and dataset not in weight_datasets):
                axis_names = ["iterations", "z", "y", "x"]
            elif ndims == 5:
                axis_names = ["iterations", "c^", "z", "y", "x"]

            data = daisy_array.data
            seg = False
            if ndims == 5 and argmax and dataset not in weight_datasets:
                data = np.argmax(np.stack([np.zeros_like(data[0]), *data]), axis=1).astype(np.uint32)
                ndims -= 1
                seg = True

            dims = neuroglancer.CoordinateSpace(
                names=axis_names,
                units="nm",
                scales=(1,) * (ndims - 3) + tuple(daisy_array.voxel_size),
            )
            voxel_offset = (0,) * (ndims - 3) + tuple(
                daisy_array.roi.offset / daisy_array.voxel_size
            )

            vol = neuroglancer.LocalVolume(
                data=data,
                voxel_offset=voxel_offset,
                dimensions=dims,
            )

            if seg:
                s.layers[dataset] = neuroglancer.SegmentationLayer(source=vol)
            else:
                s.layers[dataset] = neuroglancer.ImageLayer(source=vol, opacity=1.0)

        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=raw_datasets[::-1] + target_datasets
                        ),
                    ]
                ),
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=raw_datasets[::-1] + weight_datasets
                        ),
                    ]
                ),
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=raw_datasets[::-1] + pred_datasets
                        ),
                    ]
                ),
            ]
        )

    print(viewer)

    input("Enter to quit!")
