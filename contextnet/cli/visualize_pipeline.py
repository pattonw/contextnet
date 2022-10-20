import click


@click.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--num-voxels", type=int, default=32)
def visualize_pipeline(train_config, num_voxels):
    from contextnet.pipeline import build_pipeline, get_request, split_batch
    from contextnet.configs import TrainConfig

    from funlib.geometry import Coordinate

    import gunpowder as gp

    import neuroglancer

    import yaml

    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))
    scale_config = train_config.scale_config
    data_config = train_config.data_config

    lsds = train_config.lsds

    pipeline = build_pipeline(
        data_config,
        scale_config,
        lsds=lsds,
        sample_voxel_sizes=train_config.sample_voxel_size,
        use_organelle_datasets=train_config.use_organelle_vols,
    )

    volume_shape = Coordinate((num_voxels,) * 3)

    def load_batch(event):
        with gp.build(pipeline):
            batch = pipeline.request_batch(
                get_request(volume_shape, scale_config, lsds=lsds)
            )
        if lsds:
            raw, gt, weights, _masks, _lsds, _lsd_mask, scale = split_batch(
                batch, scale_config, lsds=lsds
            )
        else:
            raw, gt, weights, _masks, scale = split_batch(batch, scale_config, lsds=lsds)

        print(scale.data.item())

        raw_layers = []
        gt_layers = []
        weight_layers = []

        with viewer.txn() as s:
            while len(s.layers) > 0:
                del s.layers[0]

            # reverse order for raw so we can set opacity to 1, this
            # way higher res raw replaces low res when available
            for scale_level, raw_scale_array in list(enumerate(raw))[::-1]:

                dims = neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=raw_scale_array.spec.voxel_size,
                )

                raw_vol = neuroglancer.LocalVolume(
                    data=raw_scale_array.data,
                    voxel_offset=raw_scale_array.spec.roi.offset
                    / raw_scale_array.spec.voxel_size,
                    dimensions=dims,
                )

                s.layers[f"raw_s{scale_level}"] = neuroglancer.ImageLayer(
                    source=raw_vol, opacity=1.0
                )
                raw_layers.append(f"raw_s{scale_level}")
            for scale_level, gt_scale_array in enumerate(gt):
                dims = neuroglancer.CoordinateSpace(
                    names=["c^", "z", "y", "x"],
                    units="nm",
                    scales=(1,) + tuple(gt_scale_array.spec.voxel_size),
                )
                gt_vol = neuroglancer.LocalVolume(
                    data=gt_scale_array.data,
                    voxel_offset=(0,)
                    + tuple(
                        gt_scale_array.spec.roi.offset / gt_scale_array.spec.voxel_size
                    ),
                    dimensions=dims,
                )

                s.layers[f"gt_s{scale_level}"] = neuroglancer.ImageLayer(
                    source=gt_vol,
                )
                gt_layers.append(f"gt_s{scale_level}")
            for scale_level, weight_scale_array in enumerate(weights):
                dims = neuroglancer.CoordinateSpace(
                    names=["c^", "z", "y", "x"],
                    units="nm",
                    scales=(1,) + tuple(weight_scale_array.spec.voxel_size),
                )
                weight_vol = neuroglancer.LocalVolume(
                    data=weight_scale_array.data,
                    voxel_offset=(0,)
                    + tuple(
                        weight_scale_array.spec.roi.offset
                        / weight_scale_array.spec.voxel_size
                    ),
                    dimensions=dims,
                )

                s.layers[f"weight_s{scale_level}"] = neuroglancer.ImageLayer(
                    source=weight_vol,
                )
                weight_layers.append(f"weight_s{scale_level}")
            if lsds:
                raise NotImplementedError(lsds)
                for scale_level, lsd_scale_array in enumerate(_lsds):
                    dims = neuroglancer.CoordinateSpace(
                        names=["c^", "z", "y", "x"],
                        units="nm",
                        scales=(1,) + tuple(lsd_scale_array.spec.voxel_size),
                    )
                    lsd_vol = neuroglancer.LocalVolume(
                        data=lsd_scale_array.data,
                        voxel_offset=(0,)
                        + tuple(
                            lsd_scale_array.spec.roi.offset
                            / lsd_scale_array.spec.voxel_size
                        ),
                        dimensions=dims,
                    )

                    s.layers[f"lsd_s{scale_level}"] = neuroglancer.ImageLayer(
                        source=lsd_vol,
                    )
                    lsd_layers.append(f"lsd_s{scale_level}")
            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.column_layout(
                        [
                            neuroglancer.LayerGroupViewer(
                                layers=raw_layers + gt_layers
                            ),
                        ]
                    ),
                    neuroglancer.column_layout(
                        [
                            neuroglancer.LayerGroupViewer(
                                layers=raw_layers + weight_layers
                            ),
                        ]
                    ),
                ]
            )

    neuroglancer.set_server_bind_address("0.0.0.0")

    viewer = neuroglancer.Viewer()

    viewer.actions.add("load_batch", load_batch)

    with viewer.config_state.txn() as s:
        s.input_event_bindings.data_view["keyt"] = "load_batch"

    print(viewer)
    load_batch(None)

    input("Enter to quit!")
