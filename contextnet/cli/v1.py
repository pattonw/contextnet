from .train import train
from .view_snapshots import view_snapshots
from .validate import validate
from .validate_valid import validate_valid
from .visualize_pipeline import visualize_pipeline

from gunpowder.torch import Predict

import click
import yaml

import random


@click.group()
def v1():
    pass


v1.add_command(train)
v1.add_command(view_snapshots)
v1.add_command(validate)
v1.add_command(validate_valid)
v1.add_command(visualize_pipeline)


@v1.command()
@click.option("-m", "--model-config", type=click.Path(exists=True, dir_okay=False))
def model_summary(model_config):
    from contextnet.backbones.dense import DenseNet
    from contextnet.configs import BackboneConfig

    from torchsummary import summary

    model_config = BackboneConfig(**yaml.safe_load(open(model_config, "r").read()))

    in_channels = model_config.raw_input_channels + (
        model_config.n_output_channels
        if not model_config.embeddings
        else model_config.num_embeddings
    )
    print(in_channels, model_config)

    model = DenseNet(
        n_input_channels=in_channels,
        n_output_channels=model_config.n_output_channels,
        num_init_features=model_config.num_init_features,
        num_embeddings=model_config.num_embeddings,
        growth_rate=model_config.growth_rate,
        block_config=model_config.block_config,
        padding=model_config.padding,
        upsample_mode=model_config.upsample_mode,
    ).to("cpu")

    print(summary(model, (model_config.raw_input_channels, 26, 26, 26), device="cpu"))
    print(model)


@v1.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("--emb/--no-emb", type=bool, default=False)
@click.option("--argmax/--no-argmax", type=bool, default=False)
@click.option("-i", "--iteration", type=int, default=0)
def view_validations(train_config, emb, argmax, iteration):
    from contextnet.configs import TrainConfig, DataSetConfig

    import daisy

    import neuroglancer

    from sklearn.metrics import f1_score
    from scipy.special import softmax
    import numpy as np

    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))
    scale_config = train_config.scale_config
    data_config = train_config.data_config

    neuroglancer.set_server_bind_address("0.0.0.0")

    viewer = neuroglancer.Viewer()

    validation_pred_dataset = "volumes/val/{crop}/{i}/pred/scale_{scale}"
    validation_emb_dataset = "volumes/val/{crop}/{i}/emb/scale_{scale}"
    validation_raw_dataset = "volumes/val/{crop}/raw/scale_{scale}"

    datasets: list[tuple[list[str], list[str], list[str], str]] = []

    def add_layers(s, crop, gt_array):
        raw_layers = []
        pred_layers = []
        emb_layers = []
        pred_data_s0 = None
        for scale_level in range(scale_config.num_eval_scale_levels):
            if scale_level == 0:
                # pass
                continue
            # raw
            try:
                dataset = f"{crop}_raw_s{scale_level}"
                daisy_array = daisy.open_ds(
                    train_config.validation_container,
                    validation_raw_dataset.format(
                        i=iteration, crop=crop, scale=scale_level
                    ),
                )

                dims = neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=daisy_array.voxel_size,
                )

                raw_vol = neuroglancer.LocalVolume(
                    data=daisy_array.data,
                    voxel_offset=(*(daisy_array.roi.offset / daisy_array.voxel_size),),
                    dimensions=dims,
                )

                s.layers[dataset] = neuroglancer.ImageLayer(source=raw_vol, opacity=1.0)
                raw_layers = [dataset] + raw_layers
            except KeyError:
                pass

            # pred
            try:
                dataset = f"{crop}_pred_s{scale_level}"
                daisy_array = daisy.open_ds(
                    train_config.validation_container,
                    validation_pred_dataset.format(
                        i=iteration, crop=crop, scale=scale_level
                    ),
                )
                data = daisy_array.to_ndarray(daisy_array.roi)
                if scale_level == 0:
                    pred_data_s0 = data
                if train_config.lsds:
                    data = data[:-10]

                if argmax:
                    data = np.argmax(data, axis=0).astype(np.uint32)
                    dims = neuroglancer.CoordinateSpace(
                        names=["z", "y", "x"],
                        units="nm",
                        scales=daisy_array.voxel_size,
                    )
                    voxel_offset = daisy_array.roi.offset / daisy_array.voxel_size
                else:

                    dims = neuroglancer.CoordinateSpace(
                        names=["c^", "z", "y", "x"],
                        units="nm",
                        scales=(1, *daisy_array.voxel_size),
                    )
                    voxel_offset = (
                        0,
                        *(daisy_array.roi.offset / daisy_array.voxel_size),
                    )

                pred_vol = neuroglancer.LocalVolume(
                    data=data,
                    voxel_offset=voxel_offset,
                    dimensions=dims,
                )

                if argmax:
                    s.layers[dataset] = neuroglancer.SegmentationLayer(source=pred_vol)
                else:
                    s.layers[dataset] = neuroglancer.ImageLayer(
                        source=pred_vol, opacity=1.0
                    )
                pred_layers = [dataset] + pred_layers
            except KeyError:
                pass

            # emb
            dataset = f"{crop}_emb_s{scale_level}"
            try:
                daisy_array = daisy.open_ds(
                    train_config.validation_container,
                    validation_emb_dataset.format(
                        i=iteration, crop=crop, scale=scale_level
                    ),
                )

                dims = neuroglancer.CoordinateSpace(
                    names=["c^", "z", "y", "x"],
                    units="nm",
                    scales=(1, *daisy_array.voxel_size),
                )

                emb_vol = neuroglancer.LocalVolume(
                    data=daisy_array.data,
                    voxel_offset=(
                        0,
                        *(daisy_array.roi.offset / daisy_array.voxel_size),
                    ),
                    dimensions=dims,
                )

                s.layers[dataset] = neuroglancer.ImageLayer(source=emb_vol, opacity=1.0)
                emb_layers = [dataset] + emb_layers
            except KeyError:
                pass

        dataset = f"{crop}_gt"
        # compare prediction s0 to gt
        gt_data = gt_array.to_ndarray(gt_array.roi)
        for label, label_ids in enumerate(data_config.categories):
            label_data = np.isin(gt_data, label_ids)

            if pred_data_s0 is not None:
                val_score = f1_score(
                    label_data.flatten(),
                    pred_data_s0[label].flatten() > 0.5,
                )
                print(f"{label} f1 Scores: {val_score}")

        dims = neuroglancer.CoordinateSpace(
            names=["z", "y", "x"],
            units="nm",
            scales=gt_array.voxel_size,
        )

        gt_vol = neuroglancer.LocalVolume(
            data=label_data,
            voxel_offset=gt_array.roi.offset / gt_array.voxel_size,
            dimensions=dims,
        )

        s.layers[dataset] = neuroglancer.SegmentationLayer(source=gt_vol)

        return (raw_layers, pred_layers, emb_layers, dataset)

    num_datasets = len(datasets)
    global current_ind
    current_ind = 0

    with viewer.txn() as s:
        for dataset_config_path in data_config.datasets:
            dataset_config = DataSetConfig(
                **yaml.safe_load(dataset_config_path.open("r").read())
            )
            store_config = dataset_config.raw
            for validation_crop in dataset_config.validation:
                if not validation_crop in [155]:
                    # print(f"skipping crop {validation_crop} ({dataset_config.name})")
                    # continue
                    pass
                try:
                    try:
                        gt_array = daisy.open_ds(
                            store_config.container.format(dataset=dataset_config.name),
                            store_config.crop.format(
                                crop_num=validation_crop, organelle="all"
                            ),
                        )
                    except (FileNotFoundError, KeyError):
                        gt_array = daisy.open_ds(
                            store_config.fallback.format(dataset=dataset_config.name),
                            store_config.crop.format(
                                crop_num=validation_crop, organelle="all"
                            ),
                        )
                except KeyError:
                    print(f"Skipping crop {validation_crop}")
                    continue

                print(f"Adding layers for {validation_crop}")
                raw, pred, emb, gt = add_layers(s, validation_crop, gt_array)
                datasets.append((raw, pred, emb, gt))

        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=[d for dataset in datasets for d in dataset[0]]
                            + [dataset[3] for dataset in datasets]
                        ),
                    ]
                ),
                neuroglancer.column_layout(
                    [
                        neuroglancer.LayerGroupViewer(
                            layers=[d for dataset in datasets for d in dataset[0]]
                            + [d for dataset in datasets for d in dataset[1]]
                        ),
                    ]
                ),
            ]
        )

    # viewer.actions.add("switch_dataset", set_layout)

    # with viewer.config_state.txn() as s:
    #     s.input_event_bindings.data_view["keyt"] = "swith_dataset"

    print(viewer)

    input("Enter to quit!")


@v1.command()
@click.option("-t", "--train-config", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "--num-iter", type=int, default=1000)
@click.option("-o", "--output", type=click.Path(exists=False))
@click.option("-s", "--smooth", type=float, default=0)
def plot_loss(train_config, num_iter, output, smooth):
    from contextnet.configs import TrainConfig

    import numpy as np
    import matplotlib.pyplot as plt

    def smooth_func(scalars, weight):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val

        return smoothed

    train_config = TrainConfig(**yaml.safe_load(open(train_config, "r").read()))

    losses = [
        tuple(float(loss) for loss in line.strip("[]()\n").split(",") if len(loss) > 0)
        for line in list(train_config.loss_file.open().readlines())[-num_iter:]
    ]
    loss_resolutions = [np.array(loss_resolution) for loss_resolution in zip(*losses)]

    for loss_resolution in loss_resolutions:
        plt.plot(smooth_func(loss_resolution, smooth))
    plt.savefig(f"{output}.png")
