import click
import yaml


@click.command()
@click.option("-s", "--scale-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-d", "--data-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--num-voxels", type=int, default=32)
@click.option("--lsds/--no-lsds", type=bool, default=False)
def visualize_pipeline(scale_config, data_config, num_voxels, lsds):
    from contextnet.pipelines import build_pipeline, get_request, split_batch
    from contextnet.configs import ScaleConfig, DataConfig

    from funlib.geometry import Coordinate
    import gunpowder as gp

    import neuroglancer

    import numpy as np

    scale_config = ScaleConfig(**yaml.safe_load(open(scale_config, "r").read()))
    data_config = DataConfig(**yaml.safe_load(open(data_config, "r").read()))
    pipeline = (
        build_pipeline(
            data_config,
            scale_config,
            gt_voxel_size=data_config.gt_voxel_size,
            lsds=lsds,
        )
        + gp.PrintProfilingStats()
    )

    volume_shape = Coordinate((num_voxels,) * 3)

    def load_batch(event):
        print("fetching batch")
        with gp.build(pipeline):
            batch = pipeline.request_batch(
                get_request(volume_shape, scale_config, lsds=lsds)
            )
        print("Got batch")
        if lsds:
            raw, gt, weights, _lsds, _lsd_mask = split_batch(
                batch, scale_config, lsds=lsds
            )
            raw_names, gt_names, weight_names, lsd_names = (
                ["raw_high", "raw_low", "raw_context"],
                ["gt_high", "gt_low"],
                ["weight_high", "weight_low"],
                ["lsd_high", "lsd_low"],
            )
        else:
            raw, gt, weights = split_batch(batch, scale_config, lsds=lsds)
            raw_names, gt_names, weight_names = (
                ["raw_high", "raw_low", "raw_context"],
                ["gt_high", "gt_low"],
                ["weight_high", "weight_low"],
            )

        with viewer.txn() as s:
            while len(s.layers) > 0:
                del s.layers[0]

            # reverse order for raw so we can set opacity to 1, this
            # way higher res raw replaces low res when available
            for name, raw_scale_array in zip(raw_names, raw):

                dims = neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=raw_scale_array.spec.voxel_size,
                )

                raw_vol = neuroglancer.LocalVolume(
                    data=raw_scale_array.data,
                    voxel_offset=(
                        (-raw_scale_array.spec.roi.shape / 2)
                        / raw_scale_array.spec.voxel_size
                    ),
                    dimensions=dims,
                )

                s.layers[name] = neuroglancer.ImageLayer(source=raw_vol, opacity=1.0)
            for name, gt_scale_array in zip(gt_names, gt):
                dims = neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=gt_scale_array.spec.voxel_size,
                )
                gt_vol = neuroglancer.LocalVolume(
                    data=np.argmax(gt_scale_array.data, axis=0).astype(np.uint32),
                    voxel_offset=(-gt_scale_array.spec.roi.shape / 2)
                    / gt_scale_array.spec.voxel_size,
                    dimensions=dims,
                )

                s.layers[name] = neuroglancer.SegmentationLayer(
                    source=gt_vol,
                )
            for name, weight_scale_array in zip(weight_names, weights):
                dims = neuroglancer.CoordinateSpace(
                    names=["c^", "z", "y", "x"],
                    units="nm",
                    scales=(1,) + tuple(weight_scale_array.spec.voxel_size),
                )
                gt_vol = neuroglancer.LocalVolume(
                    data=weight_scale_array.data,
                    voxel_offset=(0,)
                    + tuple(
                        (-weight_scale_array.spec.roi.shape / 2)
                        / weight_scale_array.spec.voxel_size
                    ),
                    dimensions=dims,
                )

                s.layers[name] = neuroglancer.ImageLayer(
                    source=gt_vol,
                )
            if lsds:
                for name, lsd_scale_array in zip(lsd_names, _lsds):
                    dims = neuroglancer.CoordinateSpace(
                        names=["c^", "z", "y", "x"],
                        units="nm",
                        scales=(1,) + tuple(lsd_scale_array.spec.voxel_size),
                    )
                    lsd_vol = neuroglancer.LocalVolume(
                        data=lsd_scale_array.data,
                        voxel_offset=(0,)
                        + tuple(
                            (-lsd_scale_array.spec.roi.shape / 2)
                            / lsd_scale_array.spec.voxel_size
                        ),
                        dimensions=dims,
                    )

                    s.layers[name] = neuroglancer.ImageLayer(
                        source=lsd_vol,
                    )
            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.column_layout(
                        [
                            neuroglancer.LayerGroupViewer(
                                layers=raw_names[::-1] + gt_names[::-1]
                            ),
                        ]
                    ),
                    neuroglancer.column_layout(
                        [
                            neuroglancer.LayerGroupViewer(layers=raw_names[::-1] + weight_names[::-1]),
                        ]
                    ),
                    neuroglancer.column_layout(
                        [
                            neuroglancer.LayerGroupViewer(layers=raw_names[::-1]),
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
