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

    from scipy.special import softmax
    import numpy as np
    import yaml

    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))
    scale_config = train_config.scale_config

    neuroglancer.set_server_bind_address("0.0.0.0")

    viewer = neuroglancer.Viewer()

    raw_datasets = [
        f"raw_s{scale_level}"
        for scale_level in range(scale_config.num_raw_scale_levels)
    ]
    target_datasets = [
        f"target_s{scale_level}"
        for scale_level in range(scale_config.num_gt_scale_levels)
    ]
    pred_datasets = [
        f"pred_s{scale_level}"
        for scale_level in range(scale_config.num_raw_scale_levels)
    ]
    weight_datasets = [
        f"weight_s{scale_level}"
        for scale_level in range(scale_config.num_gt_scale_levels)
    ]
    loss_datasets = [
        f"loss_s{scale_level}"
        for scale_level in range(scale_config.num_gt_scale_levels)
    ]

    with viewer.txn() as s:
        while len(s.layers) > 0:
            del s.layers[0]
        for raw_dataset in raw_datasets:
            daisy_array = daisy.open_ds(
                train_config.snapshot_container,
                f"{raw_dataset}",
            )

            dims = neuroglancer.CoordinateSpace(
                names=["iterations", "z", "y", "x"],
                units="nm",
                scales=(1, *daisy_array.voxel_size),
            )

            raw_vol = neuroglancer.LocalVolume(
                data=daisy_array.data,
                voxel_offset=(
                    0,
                    *(daisy_array.roi.offset / daisy_array.voxel_size),
                ),
                dimensions=dims,
            )

            s.layers[raw_dataset] = neuroglancer.ImageLayer(source=raw_vol, opacity=1.0)
        for target_dataset in target_datasets:
            daisy_array = daisy.open_ds(
                train_config.snapshot_container,
                f"{target_dataset}",
            )

            dims = neuroglancer.CoordinateSpace(
                names=["iterations", "c^", "z", "y", "x"],
                units="nm",
                scales=(1, 1, *daisy_array.voxel_size),
            )

            target_vol = neuroglancer.LocalVolume(
                data=daisy_array.data,
                voxel_offset=(
                    0,
                    0,
                    *(daisy_array.roi.offset / daisy_array.voxel_size),
                ),
                dimensions=dims,
            )

            s.layers[target_dataset] = neuroglancer.SegmentationLayer(source=target_vol)
        for pred_dataset in pred_datasets:
            daisy_array = daisy.open_ds(
                train_config.snapshot_container,
                f"{pred_dataset}",
            )
            data = daisy_array.data
            if argmax:
                data = np.argmax(data, axis=1).astype(np.uint32)

                dims = neuroglancer.CoordinateSpace(
                    names=["iterations", "z", "y", "x"],
                    units="nm",
                    scales=(1, *daisy_array.voxel_size),
                )
                voxel_offset = (
                    0,
                    *(daisy_array.roi.offset / daisy_array.voxel_size),
                )
            else:

                dims = neuroglancer.CoordinateSpace(
                    names=["iterations", "c^", "z", "y", "x"],
                    units="nm",
                    scales=(1, 1, *daisy_array.voxel_size),
                )
                voxel_offset = (
                    0,
                    0,
                    *(daisy_array.roi.offset / daisy_array.voxel_size),
                )

            pred_vol = neuroglancer.LocalVolume(
                data=data,
                voxel_offset=voxel_offset,
                dimensions=dims,
            )
            if argmax:
                s.layers[pred_dataset] = neuroglancer.SegmentationLayer(
                    source=pred_vol,
                )
            else:
                s.layers[pred_dataset] = neuroglancer.ImageLayer(
                    source=pred_vol,
                )

        if weights:
            for weight_dataset in weight_datasets:
                daisy_array = daisy.open_ds(
                    train_config.snapshot_container,
                    f"{weight_dataset}",
                )

                dims = neuroglancer.CoordinateSpace(
                    names=["iterations", "z", "y", "x"],
                    units="nm",
                    scales=(1, *daisy_array.voxel_size),
                )

                target_vol = neuroglancer.LocalVolume(
                    data=daisy_array.data,
                    voxel_offset=(
                        0,
                        *(daisy_array.roi.offset / daisy_array.voxel_size),
                    ),
                    dimensions=dims,
                )

                s.layers[weight_dataset] = neuroglancer.ImageLayer(source=target_vol)
        if loss:
            for loss_dataset in loss_datasets:
                daisy_array = daisy.open_ds(
                    train_config.snapshot_container,
                    f"{loss_dataset}",
                )

                dims = neuroglancer.CoordinateSpace(
                    names=["iterations", "z", "y", "x"],
                    units="nm",
                    scales=(1, *daisy_array.voxel_size),
                )

                target_vol = neuroglancer.LocalVolume(
                    data=daisy_array.data,
                    voxel_offset=(
                        0,
                        *(daisy_array.roi.offset / daisy_array.voxel_size),
                    ),
                    dimensions=dims,
                )

                s.layers[loss_dataset] = neuroglancer.ImageLayer(source=target_vol)

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
                            layers=raw_datasets[::-1]
                            + pred_datasets
                            + (loss_datasets if loss else [])
                        ),
                    ]
                ),
            ]
        )

    print(viewer)

    input("Enter to quit!")